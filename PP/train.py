import math
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
# parser.add_argument('--reduction_degree', type=int, default=0.25, help='Remaining percentage of Arcs in Graph')
parser.add_argument('--data_size', type=int, default=3000, help='No. of training instances')
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
parser.add_argument('--EPOCHS', type=int, default=100,
                    help='epochs to train')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=3.5,
                    help='temperature for adj matrix')
parser.add_argument('--stepsize', type=int, default=10,
                    help='step size')
parser.add_argument('--C1', type=float, default=1, help='loss score weight')
parser.add_argument('--C2', type=float, default=10, help='penalty for depot return')
parser.add_argument('--C3', type=float, default=10, help='penalty for diagonal elements')
parser.add_argument('--C4', type=float, default=0.5, help='penalty for not visiting other nodes')

args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# torch.cuda.manual_seed(args.seed)
# np.random.seed(args.seed)

### load train instance
LENGDATA = args.data_size
problem_size = args.num_of_nodes
coords, demands, time_windows, duals, service_times, travel_times, prices = get_random_problems(LENGDATA, problem_size)

NumofTestSample = LENGDATA

dataset_scale = 1
total_samples = int(np.floor(LENGDATA * dataset_scale))

# neighborhood = 10
TC = torch.zeros((LENGDATA, args.num_of_nodes + 1, args.num_of_nodes + 1))

Price_Adj = torch.zeros((LENGDATA, args.num_of_nodes + 1, args.num_of_nodes + 1))
TC_Adj = torch.zeros((LENGDATA, args.num_of_nodes + 1, args.num_of_nodes + 1))
Targets = torch.zeros((LENGDATA, args.num_of_nodes + 1, args.num_of_nodes + 1))

for x in range(LENGDATA):
    TC[x] = calculate_compatibility(time_windows[x], travel_times[x], service_times[x])[1]
    disc_price_neg = prices[x] * torch.exp(-1 * TC[x] - torch.reshape(demands[x], (1, len(demands[x]))))
    Price_Adj[x, prices[x] < 0] = disc_price_neg[prices[x] < 0]
    disc_price_pos = prices[x] * torch.exp(TC[x] + torch.reshape(demands[x], (1, len(demands[x]))))
    Price_Adj[x, prices[x] > 0] = disc_price_pos[prices[x] > 0]
    Price_Adj[x, TC[x] == math.inf] = 0
    TC_Adj[x, TC[x] == math.inf] = 1
    # BE2(disc_price, args.reduction_degree)
    # cheapest_neighbors = torch.argsort(disc_price, dim=1)[:, :neighborhood]
    # for j in range(args.num_of_nodes + 1):
    # Price_Adj[x, j, cheapest_neighbors[j]] = disc_price[j, cheapest_neighbors[j]]
    # TC_Adj[x, j, cheapest_neighbors[j]] = TC[x, j, cheapest_neighbors[j]]

    print(x)

from models import GNN

# scattering model
model = GNN(input_dim=7, hidden_dim=args.hidden, output_dim=args.num_of_nodes + 1, n_layers=args.nlayers)

### count model parameters
print('Total number of parameters:')
print(count_parameters(model))


class PP_Dataset(Dataset):
    def __init__(self, co, tw, dul, st, dems, t_matrix, p_matrix, price_adj, tc_adj):
        self.coord = torch.FloatTensor(co)
        self.time_windows = torch.FloatTensor(tw)
        self.duals = torch.FloatTensor(dul)
        self.travel_times = torch.FloatTensor(t_matrix)
        self.prices = torch.FloatTensor(p_matrix)
        self.service_times = torch.FloatTensor(st)
        self.demands = torch.FloatTensor(dems)
        self.price_adj = price_adj
        self.tc_adj = tc_adj

    def __getitem__(self, index):
        xy_pos = self.coord[index]
        tw = self.time_windows[index]
        dual = self.duals[index]
        t_matrix = self.travel_times[index]
        p_matrix = self.prices[index]
        st = self.service_times[index]
        dems = self.demands[index]
        price_adj = self.price_adj[index]
        tc_adj = self.tc_adj[index]

        return tuple(zip(xy_pos, tw, dual, st, dems, t_matrix, p_matrix, price_adj, tc_adj))

    def __len__(self):
        return len(self.coord)


dataset = PP_Dataset(coords, time_windows, duals, service_times, demands, travel_times, prices,
                     Price_Adj, TC_Adj)

num_trainpoints = LENGDATA
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
        p_matrix = batch[6].cpu()
        price_adj_mat = batch[7].cpu()
        tc_adj_mat = batch[8].cpu()

        f0 = cor  # [:, 1:, :]
        f1 = dul  # [:, 1:]
        f2 = tw  # [:, 1:, :]
        f3 = dems  # [:, 1:]
        f4 = sts  # [:, 1:]
        # f5 = p_matrix[:, 0, 1:]
        # f6 = p_matrix[:, 1:, 0]

        dims = f1.shape
        f1 = torch.reshape(f1, (dims[0], dims[1], 1))
        f3 = torch.reshape(f3, (dims[0], dims[1], 1))
        f4 = torch.reshape(f4, (dims[0], dims[1], 1))
        # f5 = torch.reshape(f5, (dims[0], dims[1], 1))
        # f6 = torch.reshape(f6, (dims[0], dims[1], 1))

        X = torch.cat([f0, f1, f2, f3, f4], dim=2)

        distance_m = p_matrix  # [:, 1:, 1:]
        adj = torch.exp(-1. * distance_m / args.temperature)
        output = model(X, adj)

        # SEE code_blocks.py
        loss = torch.zeros(len(batch[0]))
        depot_penalties = torch.zeros(len(batch[0]))
        infeas_penalties = torch.zeros(len(batch[0]))
        visit_penalties = torch.zeros(len(batch[0]))

        for x in range(len(batch[0])):
            # SEE code_block.py

            loss[x] = torch.sum(output[x] * price_adj_mat[x])

            # pro_dist = torch.cat((torch.tensor([1]), output[x, :, 0]))
            # mat_prod = torch.matmul(torch.reshape(pro_dist, (1, args.num_of_nodes + 1)), p_mat_adj[x])
            # loss[x] = torch.matmul(mat_prod, torch.reshape(pro_dist, (args.num_of_nodes + 1, 1)))

            depot_penalties[x] = torch.maximum(1 - torch.sum(output[x, :, 0]), torch.tensor(0))
            # select_penalties[x] = torch.square(torch.sum(output[x, :, 0]) - args.reduction_size)

            infeas_penalties[x] = torch.sum(torch.diag(output[x]))  # + torch.sum(tc_adj_mat[x]*output[x])
            visit_penalties[x] = torch.sum(
                torch.maximum(dul[x] / torch.max(dul[x]) - torch.sum(output[x], dim=0), torch.zeros(dims[1])))

        batchloss = torch.sum(args.C1 * loss + args.C2 * depot_penalties + args.C3 *
                              infeas_penalties + args.C4 * visit_penalties) / len(batch[0])
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
