from ESPRCTWProblemDef import get_random_problems
from PP_exact_solver import Subproblem
from utils import *
from models import GNN

import random
import numpy as np
import argparse
import torch
import pickle
import statistics


def evaluate_reduction(LENGDATA, num_customers, model, temperature, reduction_size, MIP):
    depot_xy, coords, demands, time_windows, depot_time_window, duals, service_times, travel_times, prices = \
        get_random_problems(LENGDATA, num_customers)
    coords, time_windows, demands, service_times, duals = merge_with_depot(depot_xy, coords,
                                                                           demands, time_windows, depot_time_window,
                                                                           duals,
                                                                           service_times)

    f0 = coords[:, 1:, :]
    f1 = time_windows[:, 1:, :]
    f2 = duals[:, 1:]
    dims = f2.shape
    f2 = torch.reshape(f2, (dims[0], dims[1], 1))
    X = torch.cat([f0, f1, f2], dim=2)
    distance_m = travel_times.cpu()[:, 1:, 1:]
    adj = torch.exp(-1. * distance_m / temperature)
    output = model(X, adj)
    sorted_indices = output.argsort(dim=1, descending=True)[:, :reduction_size, 0]
    Scores = []
    print("Reduction Made")

    for x in range(LENGDATA):

        NR = Node_Reduction(None, coords[x])
        red_cor = NR.dual_based_elimination(duals[x])
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

        print(ordered_route)
        print(reduced_cost)
        print("subproblem solved")

        miscount = 0
        for node in ordered_route:
            if node != 0 and node not in sorted_indices[x]:
                miscount += 1

        print("The length of the optimal route is: " + str(len(ordered_route)))
        print("The number of missing customers is: " + str(miscount))
        print("----------------")
        print("")
        Scores.append((miscount / len(ordered_route)))

    print("On average, " + str(round(statistics.mean(Scores), 2) * 100) + "% of nodes are missing. ")
    pickle_out = open('Reduction Results for N=' + str(num_customers) + " with reduction size " + str(reduction_size),
                      'wb')
    pickle.dump(Scores, pickle_out)
    pickle_out.close()


def main():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=40)
    parser.add_argument('--data_size', type=int, default=10, help='No. of evaluation instances')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--reduction_size', type=int, default=20, help='Remaining Nodes in Graph')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--temperature', type=float, default=3.5, help='temperature for adj matrix')
    parser.add_argument('--nlayers', type=int, default=2, help='num of layers')
    parser.add_argument('--MIP', type=bool, default=True, help='use MIP for exact method.')

    args = parser.parse_args()
    num_customers = args.num_customers
    device = args.device
    LENGDATA = args.data_size
    MIP = args.MIP

    print("Instances are of size " + str(args.num_customers))

    model = GNN(input_dim=5, hidden_dim=args.hidden, output_dim=2, n_layers=args.nlayers)
    model_name = 'Saved_Models/PP_%d/scatgnn_layer_2_hid_%d_model_290_temp_3.500.pth' % (
    args.num_customers, args.hidden)
    checkpoint_main = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint_main)
    evaluate_reduction(LENGDATA, num_customers, model, args.temperature, args.reduction_size, MIP)


if __name__ == "__main__":
    main()
