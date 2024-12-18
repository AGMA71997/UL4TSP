from ESPRCTWProblemDef import get_random_problems
from ESPRCTWEnv import ESPRCTWEnv as Env
from ESPRCTWModel import ESPRCTWModel as Model
from PP_exact_solver import Subproblem
from utils import *
from models import GNN
from ESPPRC_heuristic import DSSR_ESPPRC, ESPPRC

import random
import numpy as np
import argparse
import torch
import pickle
import statistics


def evaluate_reduction(LENGDATA, num_customers, model, temperature, MIP):

    coords, demands, time_windows, duals, service_times, travel_times, prices = \
        get_random_problems(LENGDATA, num_customers)

    f0 = coords  # [:, 1:, :]
    f1 = duals  # [:, 1:]
    f2 = time_windows  # [:, 1:, :]
    f3 = demands  # [:, 1:]
    f4 = service_times  # [:, 1:]
    # f5 = p_matrix[:, 0, 1:]
    # f6 = p_matrix[:, 1:, 0]

    dims = f1.shape
    f1 = torch.reshape(f1, (dims[0], dims[1], 1))
    f3 = torch.reshape(f3, (dims[0], dims[1], 1))
    f4 = torch.reshape(f4, (dims[0], dims[1], 1))
    # f5 = torch.reshape(f5, (dims[0], dims[1], 1))
    # f6 = torch.reshape(f6, (dims[0], dims[1], 1))

    X = torch.cat([f0, f1, f2, f3, f4], dim=2)

    mask = torch.ones(num_customers + 1, num_customers + 1).cpu()
    mask.fill_diagonal_(0)

    TC = torch.zeros((LENGDATA, num_customers + 1, num_customers + 1))

    Price_Adj = torch.zeros((LENGDATA, num_customers + 1, num_customers + 1))
    for x in range(LENGDATA):
        TC[x] = calculate_compatibility(time_windows[x], travel_times[x], service_times[x])[1]
        disc_price_neg = prices[x] * torch.exp(-1 * TC[x] - torch.reshape(demands[x], (1, len(demands[x]))))
        Price_Adj[x, prices[x] < 0] = disc_price_neg[prices[x] < 0]
        disc_price_pos = prices[x] * torch.exp(TC[x] + torch.reshape(demands[x], (1, len(demands[x]))))
        Price_Adj[x, prices[x] > 0] = disc_price_pos[prices[x] > 0]
        Price_Adj[x, TC[x] == math.inf] = 2
        print(x)

    distance_m = Price_Adj  # [:, 1:, 1:]
    adj = torch.exp(-1. * distance_m / temperature)
    adj *= mask
    output = model(X, adj)

    point_wise_distance = torch.matmul(output, torch.roll(torch.transpose(output, 1, 2), -1, 1))

    print("Reduction Made")
    for x in range(LENGDATA):
        k = 10
        tensor = point_wise_distance[x]  # Price_Adj[x] # output[x] #

        for col_idx in range(tensor.shape[1]):
            # Extract the column
            col = tensor[:, col_idx]

            # Find the indices of the k smallest elements
            smallest_indices = torch.topk(col, k, largest=True).indices

            col = Price_Adj[x, :, col_idx]
            # Extract the k smallest elements
            smallest_values = col[smallest_indices]
            print((col_idx, smallest_values.tolist(), smallest_indices.tolist()))

        print("################")

        if MIP:
            if x >= 0:
                continue
            subproblem = Subproblem(num_customers, 1, travel_times, demands, time_windows,
                                    service_times, None, [])
            ordered_route, reduced_cost = subproblem.MIP_program()
            if len(ordered_route) == 3 and reduced_cost >= 0:
                ordered_route = []
                reduced_cost = 0
        else:
            algo = ESPPRC(1, demands, time_windows, service_times, num_customers,
                          travel_times, prices)
            opt_labels = algo.solve()
            print(len(opt_labels))
            if len(opt_labels) > 0:
                ordered_route = opt_labels[0].path()
                reduced_cost = opt_labels[0].cost
            else:
                ordered_route = []
                reduced_cost = 0

        # HERE

        '''print("The route with a reduced instance: " + str(ordered_route))
        print("with Objective: " + str(reduced_cost))'''
        # Exact.append(reduced_cost)

    # print("On average, " + str(round(statistics.mean(Scores), 2)) + " nodes are missing. ")
    # print("The mean exact score is: " + str(statistics.mean(Exact)))

    '''pickle_out = open('Reduction Results for N=' + str(num_customers) + " with reduction size " + str(reduction_size),
                      'wb')
    pickle.dump(Scores, pickle_out)
    pickle_out.close()'''


def main():
    random.seed(25)
    np.random.seed(25)
    torch.manual_seed(25)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=200)
    parser.add_argument('--data_size', type=int, default=10, help='No. of evaluation instances')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs used.')
    parser.add_argument('--temperature', type=float, default=3.5, help='temperature for adj matrix')
    parser.add_argument('--nlayers', type=int, default=2, help='num of layers')
    parser.add_argument('--MIP', type=bool, default=True, help='use MIP for exact method.')

    args = parser.parse_args()
    device = torch.device(args.device)
    num_customers = args.num_customers
    LENGDATA = args.data_size
    MIP = args.MIP

    print("Instances are of size " + str(args.num_customers))

    model = GNN(input_dim=7, hidden_dim=args.hidden, output_dim=args.num_customers + 1, n_layers=args.nlayers)
    model_name = 'Saved_Models/PP_%d/scatgnn_layer_%d_hid_%d_model_%d_temp_3.500.pth' % \
                 (args.num_customers, args.nlayers, args.hidden, args.epochs)
    checkpoint_main = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint_main)

    evaluate_reduction(LENGDATA, num_customers, model, args.temperature, MIP)


if __name__ == "__main__":
    main()
