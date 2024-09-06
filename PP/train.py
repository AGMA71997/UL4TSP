import random
import time
import torch.nn
from utils import *
from torch.utils.data import Dataset, DataLoader  # use pytorch dataloader
import numpy as np
import statistics
import argparse
from ESPRCTWProblemDef import get_random_problems

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=200, help='Graph Size')
parser.add_argument('--reduction_size', type=int, default=40, help='Remaining Nodes in Graph')
parser.add_argument('--data_size', type=int, default=2000, help='No. of training instances')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning Rate')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=2,
                    help='num of layers')
parser.add_argument('--EPOCHS', type=int, default=200,
                    help='epochs to train')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=3.5,
                    help='temperature for adj matrix')
parser.add_argument('--stepsize', type=int, default=10,
                    help='step size')
parser.add_argument('--C1', type=float, default=1, help='loss score weight')
parser.add_argument('--C2', type=float, default=10, help='penalty for over-selection')
parser.add_argument('--TW_pen', type=float, default=0, help='penalty for time windows')
parser.add_argument('--dem_pen', type=float, default=0, help='penalty for demands')
parser.add_argument('--dual_pen', type=float, default=0, help='penalty for dual')

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
model = GNN(input_dim=3, hidden_dim=args.hidden, output_dim=2, n_layers=args.nlayers)

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
        f1 = dul[:, 1:]
        # f2 = tw[:, 1:, 1] - tw[:, 1:, 0]
        # f3 = dems[:, 1:]
        dims = f1.shape
        f1 = torch.reshape(f1, (dims[0], dims[1], 1))
        # f2 = torch.reshape(f2, (dims[0], dims[1], 1))
        # f3 = torch.reshape(f3, (dims[0], dims[1], 1))
        X = torch.cat([f0, f1], dim=2)
        distance_m = p_matrix[:, 1:, 1:]

        adj = torch.exp(-1. * distance_m / args.temperature)
        output = model(X, adj)

        # SEE code_blocks.py
        loss = torch.zeros(len(batch[0]))
        select_penalties = torch.zeros(len(batch[0]))
        tw_penalties = torch.zeros(len(batch[0]))
        dem_penalties = torch.zeros(len(batch[0]))
        dual_penalties = torch.zeros(len(batch[0]))
        # indices = retain_indices(output, args.reduction_size)
        for x in range(len(batch[0])):
            # SEE code_block.py
            mat_prod = torch.matmul(torch.reshape(output[x, :, 0], (1, args.num_of_nodes)), p_matrix[x, 1:, 1:])
            loss[x] = torch.matmul(mat_prod, torch.reshape(output[x, :, 0], (args.num_of_nodes, 1)))
            select_penalties[x] = torch.square(torch.sum(output[x, :, 0]) - args.reduction_size)
            tw_penalties[x] = torch.dot(output[x, :, 0], tw[x, 1:, 1] - tw[x, 1:, 0])
            dem_penalties[x] = torch.dot(output[x, :, 0], dems[x, 1:])
            dual_penalties[x] = torch.dot(output[x, :, 0], duals[x, 1:])

        # select_penalties = torch.maximum(select_penalties, torch.zeros(select_penalties.shape))
        batchloss = torch.sum(args.C1 * loss + args.C2 * select_penalties - args.TW_pen * tw_penalties
                              + args.dem_pen * dem_penalties - args.dual_pen * dual_penalties) / len(batch[0])
        Losses.append(batchloss.item())

        print('Loss: %.5f' % batchloss.item())
        # probas = torch.sort(output, 1, descending=True)[0]
        # print(probas[0, :, 0])
        optimizer.zero_grad()
        batchloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        counter += len(batch[0])

    print("The mean loss for epoch " + str(epoch) + " is: " + str(statistics.mean(Losses)))


for i in range(1, args.EPOCHS + 1):
    train(i)
    if (i >= 2) and (i % 10 == 0):
        torch.save(model.state_dict(), 'Saved_Models/PP_%d/scatgnn_layer_%d_hid_%d_model_%d_temp_%.3f.pth' % (
            args.num_of_nodes, args.nlayers, args.hidden, i, args.temperature))
