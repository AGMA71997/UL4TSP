import torch
import torch.nn.functional as F
from torch.nn import Linear
import time
from torch import tensor
import torch.nn
from utils import *
import pickle
from torch.utils.data import Dataset, DataLoader  # use pytorch dataloader
from random import shuffle
import numpy as np
import argparse
from ESPRCTWProblemDef import get_random_problems

from ESPRCTWEnv import ESPRCTWEnv as Env
from ESPRCTWModel import ESPRCTWModel as Model

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=200, help='Graph Size')
parser.add_argument('--reduction_size', type=int, default=100, help='Remaining Nodes in Graph')
parser.add_argument('--data_size', type=int, default=3000, help='no. of training instances')
parser.add_argument('--lr', type=float, default=3e-3,
                    help='Learning Rate')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=3,
                    help='num of layers')
parser.add_argument('--EPOCHS', type=int, default=20,
                    help='epochs to train')
parser.add_argument('--penalty_coefficient', type=float, default=2.,
                    help='penalty_coefficient')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=3.5,
                    help='temperature for adj matrix')
parser.add_argument('--rescale', type=float, default=2.,
                    help='rescale for xy plane')
parser.add_argument('--C1_penalty', type=float, default=10.,
                    help='penalty row/column')
parser.add_argument('--topk', type=int, default=10,
                    help='topk')
parser.add_argument('--stepsize', type=int, default=20,
                    help='step size')
parser.add_argument('--diag_loss', type=float, default=0.1,
                    help='penalty on the diag')
args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

### load train instance
LENGDATA = args.data_size
problem_size = args.num_of_nodes
depot_xy, coords, demands, time_windows, depot_time_window, duals, service_times, travel_times, prices = \
    get_random_problems(LENGDATA, problem_size)

coords, time_windows, demands, service_times, duals = merge_with_depot(depot_xy, coords,
                                                                       demands, time_windows, depot_time_window, duals,
                                                                       service_times)

NumofTestSample = LENGDATA

dataset_scale = 1
total_samples = int(np.floor(LENGDATA * dataset_scale))

preposs_time = time.time()

from models import GNN

# scattering model
# model = GNN(input_dim=5, hidden_dim=args.hidden, output_dim=args.num_of_nodes, n_layers=args.nlayers)
model = GNN(input_dim=5, hidden_dim=args.hidden, output_dim=1, n_layers=args.nlayers)

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
mask = torch.ones(args.num_of_nodes).cpu()  # cuda()

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

model_path = 'C:/Users/abdug/Python/POMO-implementation/ESPRCTW/POMO/result/model100_scaler_max_t_data'
model_epoch = 160
model_load = {
    'path': model_path,
    'epoch': model_epoch}

env_params = {'problem_size': 100,
              'pomo_size': 100}

POMO = Model(**model_params)
device = torch.device('cpu')
checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
checkpoint = torch.load(checkpoint_fullname, map_location=device)
POMO.load_state_dict(checkpoint['model_state_dict'])


def train(epoch):
    scheduler.step()
    model.train()
    print('Epoch: %d' % epoch)
    for batch in train_loader:
        cor = batch[0].cpu()
        tw = batch[1].cpu()
        dul = batch[2].cpu()
        sts = batch[3].cpu()
        dems = batch[4].cpu()
        t_matrix = batch[5].cpu()
        p_matrix = batch[6].cpu()

        f0 = cor[:, 1:, :]  # .cuda()
        f1 = tw[:, 1:, :]
        f2 = dul[:, 1:]
        dims = f2.shape
        f2 = torch.reshape(f2, (dims[0], dims[1], 1))
        X = torch.cat([f0, f1, f2], dim=2)
        distance_m = batch[5].cpu()[:, 1:, 1:]
        adj = torch.exp(-1. * distance_m / args.temperature)
        # adj *= mask
        output = model(X, adj)
        sorted_indices = output.argsort(dim=1, descending=True)[:, :args.reduction_size]
        # print(sorted_indices[:, 0:5].reshape((len(batch[0]), 5)))
        NR = Node_Reduction(sorted_indices, cor)
        red_cor = NR.reduce_instance()
        red_cor, red_dem, red_tws, red_duals, red_sts, red_tms, red_prices, cus_mapping = reshape_problem(red_cor,
                                                                                                          dems,
                                                                                                          tw,
                                                                                                          dul,
                                                                                                          sts,
                                                                                                          t_matrix,
                                                                                                          p_matrix,
                                                                                                          args.reduction_size + 1)

        env = Env(**env_params)
        env.declare_problem(red_cor, red_dem, red_tws, red_duals, red_sts, red_tms, red_prices, 1, len(batch[0]))
        pp_rl_solver = ESPRCTW_RL_solver(env, POMO, red_prices)
        promising_columns, best_columns, best_rewards = pp_rl_solver.generate_columns()

        batchloss = torch.sum(best_rewards) / len(batch[0])

        print('Loss: %.5f' % batchloss.item())
        optimizer.zero_grad()
        batchloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


for i in range(args.EPOCHS):
    train(i)
    if (i >= 200) and (i % 10 == 0):
        torch.save(model.state_dict(), 'Saved_Models/TSP_%d/scatgnn_layer_%d_hid_%d_model_%d_temp_%.3f.pth' % (
            args.num_of_nodes, args.nlayers, args.hidden, i, args.temperature))
