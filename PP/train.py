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
parser.add_argument('--reduction_size', type=int, default=100, help='Remaining Nodes in Graph')
parser.add_argument('--data_size', type=int, default=5000, help='No. of training instances')
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
parser.add_argument('--EPOCHS', type=int, default=80,
                    help='epochs to train')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=3.5,
                    help='temperature for adj matrix')
parser.add_argument('--stepsize', type=int, default=10,
                    help='step size')
parser.add_argument('--C1', type=float, default=1, help='loss score weight')
parser.add_argument('--C2', type=float, default=10, help='penalty for over-selection')
parser.add_argument('--TC_pen', type=float, default=0, help='penalty for time consumption')
parser.add_argument('--dem_pen', type=float, default=0, help='penalty for demands')

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

neighborhood = 10
TC = torch.zeros((LENGDATA, args.num_of_nodes + 1, args.num_of_nodes + 1))

Price_Adj = torch.zeros((LENGDATA, args.num_of_nodes + 1, args.num_of_nodes + 1))
TC_Adj = torch.zeros((LENGDATA, args.num_of_nodes + 1, args.num_of_nodes + 1))

for x in range(LENGDATA):
    TC[x] = calculate_compatibility(time_windows[x], travel_times[x], service_times[x])[1]
    disc_price = prices[x]*torch.exp(-1 * TC[x] - torch.reshape(demands[x], (1, len(demands[x]))))
    cheapest_neighbors = torch.argsort(disc_price, dim=1)[:, :neighborhood]
    for j in range(args.num_of_nodes + 1):
        for k in cheapest_neighbors[j]:
            if TC[x, j, k] != math.inf and j != k:
                Price_Adj[x, j, k] = disc_price[j, k]
                TC_Adj[x, j, k] = TC[x, j, k]
    print(x)

from models import GNN

# scattering model
model = GNN(input_dim=9, hidden_dim=args.hidden, output_dim=2, n_layers=args.nlayers)

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
        p_mat_adj = batch[7].cpu()
        tc_mat_adj = batch[8].cpu()

        f0 = cor[:, 1:, :]  # .cuda()
        f1 = dul[:, 1:]
        f2 = tw[:, 1:, :]
        f3 = dems[:, 1:]
        f4 = sts[:, 1:]
        f5 = p_matrix[:, 0, 1:]
        f6 = p_matrix[:, 1:, 0]

        dims = f1.shape
        f1 = torch.reshape(f1, (dims[0], dims[1], 1))
        f3 = torch.reshape(f3, (dims[0], dims[1], 1))
        f4 = torch.reshape(f4, (dims[0], dims[1], 1))
        f5 = torch.reshape(f5, (dims[0], dims[1], 1))
        f6 = torch.reshape(f6, (dims[0], dims[1], 1))

        X = torch.cat([f0, f1, f2, f3, f4, f5, f6], dim=2)

        distance_m = p_matrix[:, 1:, 1:]
        adj = torch.exp(-1. * distance_m / args.temperature)
        output = model(X, adj)

        # SEE code_blocks.py
        loss = torch.zeros(len(batch[0]))
        select_penalties = torch.zeros(len(batch[0]))
        tc_penalties = torch.zeros(len(batch[0]))
        dem_penalties = torch.zeros(len(batch[0]))

        for x in range(len(batch[0])):
            # SEE code_block.py
            pro_dist = torch.cat((torch.tensor([1]), output[x, :, 0]))

            mat_prod = torch.matmul(torch.reshape(pro_dist, (1, args.num_of_nodes + 1)), p_mat_adj[x])
            loss[x] = torch.matmul(mat_prod, torch.reshape(pro_dist, (args.num_of_nodes + 1, 1)))

            mat_prod2 = torch.matmul(torch.reshape(pro_dist, (1, args.num_of_nodes + 1)), tc_mat_adj[x])
            tc_penalties[x] = torch.matmul(mat_prod2, torch.reshape(pro_dist, (args.num_of_nodes + 1, 1)))

            select_penalties[x] = torch.square(torch.sum(output[x, :, 0]) - args.reduction_size)
            dem_penalties[x] = torch.dot(output[x, :, 0], dems[x, 1:])

        # select_penalties = torch.maximum(select_penalties, torch.zeros(select_penalties.shape))
        batchloss = torch.sum(args.C1 * loss + args.C2 * select_penalties + args.TC_pen * tc_penalties
                              + args.dem_pen * dem_penalties) / len(batch[0])
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
