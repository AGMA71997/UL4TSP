import statistics

import torch
import torch.nn.functional as F
from torch.nn import Linear
import time
from torch import tensor
import torch.nn
import pickle
from torch.utils.data import Dataset, DataLoader  # use pytorch dataloader
from random import shuffle
import numpy as np
from utils import *
import argparse
from ESPRCTWProblemDef import get_random_problems
from ESPRCTWEnv import ESPRCTWEnv as Env
from ESPRCTWModel import ESPRCTWModel as Model

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_of_nodes', type=int, default=30, help='Graph Size')
parser.add_argument('--reduction_size', type=int, default=30, help='Remaining Nodes in Graph')
parser.add_argument('--POMO_path', type=str, default='model20_scaler2_max_t_data', help='POMO model path')
parser.add_argument('--POMO_epoch', type=int, default=200, help='POMO model epoch')
parser.add_argument('--data_size', type=int, default=10, help='No. of training instances')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='smoo')
parser.add_argument('--moment', type=int, default=1,
                    help='scattering moment')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size')
parser.add_argument('--nlayers', type=int, default=3,
                    help='num of layers')
parser.add_argument('--use_smoo', action='store_true')
parser.add_argument('--epoch', type=int, default=20,
                    help='Training epochs of loaded model')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--temperature', type=float, default=3.5,
                    help='temperature for adj matrix')
parser.add_argument('--rescale', type=float, default=1.,
                    help='rescale for xy plane')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device')
args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
device = args.device

NumofTestSample = args.data_size
problem_size = args.num_of_nodes
depot_xy, coords, demands, time_windows, depot_time_window, duals, service_times, travel_times, prices = \
    get_random_problems(NumofTestSample, problem_size)

coords, time_windows, demands, service_times, duals = merge_with_depot(depot_xy, coords,
                                                                       demands, time_windows, depot_time_window, duals,
                                                                       service_times)

total_samples = args.data_size

from models import GNN

# scattering model
model = GNN(input_dim=5, hidden_dim=args.hidden, output_dim=2, n_layers=args.nlayers)
# model = model.to(device)


### count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

testdata = dataset[0:]  ##this is very important!
TestData_size = len(testdata)
batch_size = args.batch_size
test_loader = DataLoader(testdata, batch_size, shuffle=False)

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


def test(loader):
    TestData_size = len(loader.dataset)
    scores = []
    model.eval()
    for batch in loader:
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
        f3 = dems[:, 1:]
        dims = f2.shape
        f2 = torch.reshape(f2, (dims[0], dims[1], 1))
        f3 = torch.reshape(f3, (dims[0], dims[1], 1))
        X = torch.cat([f0, f1, f2, f3], dim=2)
        distance_m = batch[5].cpu()[:, 1:, 1:]
        adj = torch.exp(-1. * distance_m / args.temperature)
        output = model(X, adj)
        sorted_indices = output.argsort(dim=1, descending=True)[:, :args.reduction_size, 0]
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
        loss = torch.sum(best_rewards) / len(batch[0])
        scores.append(loss.item())

    return scores


# PP200
model_name = 'Saved_Models/PP_%d/scatgnn_layer_%d_hid_%d_model_%d_temp_3.500.pth' % (args.num_of_nodes, args.nlayers, args.hidden, args.epoch)
checkpoint_main = torch.load(model_name, map_location=device)
model.load_state_dict(checkpoint_main)
scores = test(test_loader)
print(scores)
print(statistics.mean(scores))

print('Finish Inference!')
