import time
import torch.nn
from utils import *
from torch.utils.data import Dataset, DataLoader  # use pytorch dataloader
import numpy as np
import statistics
import argparse
from ESPRCTWProblemDef import get_random_problems
from PP_exact_solver import Subproblem

from ESPRCTWEnv import ESPRCTWEnv as Env
from ESPRCTWModel import ESPRCTWModel as Model

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=50, help='Graph Size')
parser.add_argument('--reduction_size', type=int, default=20, help='Remaining Nodes in Graph')
parser.add_argument('--POMO_path', type=str, default='model20_max_t_data_Nby2', help='POMO model path')
parser.add_argument('--POMO_epoch', type=int, default=200, help='POMO model epoch')
parser.add_argument('--data_size', type=int, default=500, help='No. of training instances')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning Rate')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=3,
                    help='num of layers')
parser.add_argument('--EPOCHS', type=int, default=100,
                    help='epochs to train')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=3.5,
                    help='temperature for adj matrix')
parser.add_argument('--stepsize', type=int, default=10,
                    help='step size')
parser.add_argument('--C1', type=float, default=1, help='loss score weight')
parser.add_argument('--C2', type=float, default=1, help='penalty for over-selection')

args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

### load train instance
LENGDATA = args.data_size
problem_size = args.num_of_nodes
coords, demands, time_windows, duals, service_times, travel_times, prices = get_random_problems(LENGDATA, problem_size)

NumofTestSample = LENGDATA

dataset_scale = 1
total_samples = int(np.floor(LENGDATA * dataset_scale))

preposs_time = time.time()

from models import GNN

# scattering model
model = GNN(input_dim=5, hidden_dim=args.hidden, output_dim=2, n_layers=args.nlayers)

### count model parameters
print('Total number of parameters:')
print(count_parameters(model))


class PP_Dataset(Dataset):
    def __init__(self, co, tw, dul, st, dems, t_matrix, p_matrix, ):
        self.coord = torch.FloatTensor(co)
        self.time_windows = torch.FloatTensor(tw)
        self.duals = torch.FloatTensor(dul)
        self.travel_times = torch.FloatTensor(t_matrix)
        self.prices = torch.FloatTensor(p_matrix)
        self.service_times = torch.FloatTensor(st)
        self.demands = torch.FloatTensor(dems)

    def __getitem__(self, index):
        xy_pos = self.coord[index]
        tw = self.time_windows[index]
        dual = self.duals[index]
        t_matrix = self.travel_times[index]
        p_matrix = self.prices[index]
        st = self.service_times[index]
        dems = self.demands[index]

        return tuple(zip(xy_pos, tw, dual, st, dems, t_matrix, p_matrix))

    def __len__(self):
        return len(self.coord)


dataset = PP_Dataset(coords, time_windows, duals, service_times, demands, travel_times, prices)

num_trainpoints = 2000
num_valpoints = total_samples - num_trainpoints
sctdataset = dataset
traindata = sctdataset[0:num_trainpoints]
valdata = sctdataset[num_trainpoints:]
batch_size = args.batch_size
train_loader = DataLoader(traindata, batch_size, shuffle=True)
val_loader = DataLoader(valdata, batch_size, shuffle=False)

from torch.optim.lr_scheduler import StepLR

# optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.wdecay)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
scheduler = StepLR(optimizer, step_size=args.stepsize, gamma=0.8)
model.cpu()  # cuda()

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

model_path = args.POMO_path
model_epoch = args.POMO_epoch
model_load = {
    'path': model_path,
    'epoch': model_epoch}

env_params = {'problem_size': args.reduction_size,
              'pomo_size': args.reduction_size}

POMO = Model(**model_params)
device = torch.device('cpu')
checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
checkpoint = torch.load(checkpoint_fullname, map_location=device)
POMO.load_state_dict(checkpoint['model_state_dict'])


def train(epoch):
    scheduler.step()
    model.train()
    print('Epoch: %d' % epoch)
    Losses = []
    counter = 0
    for batch in train_loader:
        cor = batch[0].cpu()
        tw = batch[1].cpu()
        dul = batch[2].cpu()
        sts = batch[3].cpu()
        dems = batch[4].cpu()
        t_matrix = batch[5].cpu()
        p_matrix = batch[6].cpu()

        f0 = cor[:, 1:, :]  # .cuda()
        f1 = tw[:, 1:, 1] - tw[:, 1:, 0]
        f2 = dul[:, 1:]
        f3 = dems[:, 1:]
        dims = f2.shape
        f1 = torch.reshape(f1, (dims[0], dims[1], 1))
        f2 = torch.reshape(f2, (dims[0], dims[1], 1))
        f3 = torch.reshape(f3, (dims[0], dims[1], 1))
        X = torch.cat([f0, f1, f2, f3], dim=2)
        distance_m = t_matrix[:, 1:, 1:]
        adj = torch.exp(-1. * distance_m / args.temperature)
        output = model(X, adj)

        if counter not in Instance_Baselines:
            for inst in range(len(batch[0])):
                Excludes[counter + inst] = [j - 1 for j in range(1, len(dul[inst])) if dul[inst, j] == 0]
                env = Env(**{'problem_size': args.num_of_nodes, 'pomo_size': args.num_of_nodes})
                env.declare_problem(cor, dems, tw, dul, sts, t_matrix, p_matrix, len(batch[0]))
                pp_rl_solver = ESPRCTW_RL_solver(env, POMO)
                _, _, baseloss, _ = pp_rl_solver.get_loss()
                Instance_Baselines[counter + inst] = baseloss[inst]

        baselines = {}
        exclus = {}
        for inst in range(len(batch[0])):
            baselines[inst] = Instance_Baselines[counter + inst]
            exclus[inst] = Excludes[counter + inst]

        indices = retain_indices(output, args.reduction_size)
        loss = torch.zeros(len(batch[0]))
        penalties = torch.zeros(len(batch[0]))
        for x in range(len(batch[0])):
            NR = Node_Reduction(indices[x:x + 1], coords[x:x + 1])
            red_cor = NR.reduce_instance()[0]
            red_cor, red_dem, red_tws, red_sts, red_tms, red_prices, cus_mapping = reshape_problem_2(red_cor,
                                                                                                     demands[x],
                                                                                                     time_windows[x],
                                                                                                     service_times[x],
                                                                                                     travel_times[x],
                                                                                                     prices[x])

            N = len(red_cor) - 1
            subproblem = Subproblem(N, 1, red_tms, red_dem, red_tws, red_sts, red_prices, [])

            if MIP:
                ordered_route, reduced_cost = subproblem.MIP_program()
            else:
                ordered_route, reduced_cost, top_labels = subproblem.solve()

            ordered_route = remap_route(ordered_route, cus_mapping)

            penalties[x] = torch.sum(output[x, :, 0]) - args.reduction_size
            penalties = torch.maximum(penalties, torch.zeros(penalties.shape))
            factor = reduced_cost - baselines[x]
            for node in ordered_route:
                if node != 0:
                    loss[x] = output[x, node - 1, 0] * - math.exp(dul[x, node] + abs(factor))

        batchloss = torch.sum(args.C1 * loss + args.C2 * penalties) / len(batch[0])

        Losses.append(batchloss.item())

        print('Loss: %.5f' % batchloss.item())
        optimizer.zero_grad()
        batchloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        counter += len(batch[0])

    print("The mean loss for epoch " + str(epoch) + " is: " + str(statistics.mean(Losses)))


Instance_Baselines = {}
Excludes = {}
MIP = True
for i in range(1, args.EPOCHS + 1):
    train(i)
    if (i >= 2) and (i % 10 == 0):
        torch.save(model.state_dict(), 'Saved_Models/PP_%d/scatgnn_layer_%d_hid_%d_model_%d_temp_%.3f.pth' % (
            args.num_of_nodes, args.nlayers, args.hidden, i, args.temperature))
