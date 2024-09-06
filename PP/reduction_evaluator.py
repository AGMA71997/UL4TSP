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


def evaluate_reduction(LENGDATA, num_customers, model, temperature, reduction_size, MIP, env_params, POMO):
    coords, demands, time_windows, duals, service_times, travel_times, prices = \
        get_random_problems(LENGDATA, num_customers)

    f0 = coords[:, 1:, :]
    f1 = duals[:, 1:]
    # f2 = time_windows[:, 1:, 1]-time_windows[:, 1:, 0]
    # f3 = demands[:, 1:]
    dims = f1.shape
    f1 = torch.reshape(f1, (dims[0], dims[1], 1))
    # f2 = torch.reshape(f2, (dims[0], dims[1], 1))
    # f3 = torch.reshape(f3, (dims[0], dims[1], 1))
    X = torch.cat([f0, f1], dim=2)
    distance_m = prices[:, 1:, 1:]
    adj = torch.exp(-1. * distance_m / temperature)
    output = model(X, adj)
    probas = torch.sort(output, 1, descending=True)[0]
    print(probas[0, :, 0])
    print("Reduction Made")

    sorted_indices = retain_indices(output, reduction_size)

    env = Env(**env_params)
    env.declare_problem(coords, demands, time_windows, duals, service_times, travel_times,
                        prices * -1, LENGDATA)
    pp_rl_solver = ESPRCTW_RL_solver(env, POMO)
    loss, best_columns2, best_rewards2, penalties = pp_rl_solver.get_loss()

    Scores = []
    # POMO_ML = []
    POMO_Dual = []
    Exact = []
    for x in range(LENGDATA):
        '''print(statistics.mean(duals[x].tolist()))
        print(sorted(duals[x].tolist(), reverse=True))
        print(max(duals[x]))
        non0 = [d for d in duals[x] if d > 0]
        print(min(non0))
        print(len(non0))
        print("--------------")'''

        NR = Node_Reduction(sorted_indices[x:x + 1], coords[x:x + 1])
        red_cor = NR.reduce_instance()[0]
        red_cor, red_dem, red_tws, red_sts, red_tms, red_prices, cus_mapping = reshape_problem_2(red_cor,
                                                                                                 demands[x],
                                                                                                 time_windows[x],
                                                                                                 service_times[x],
                                                                                                 travel_times[x],
                                                                                                 prices[x])

        N = len(red_cor) - 1

        if MIP:
            subproblem = Subproblem(N, 1, red_tms, red_dem, red_tws, red_sts, red_prices, [])
            ordered_route, reduced_cost = subproblem.MIP_program()
            if len(ordered_route) == 3 and reduced_cost >= 0:
                ordered_route = []
                reduced_cost = 0
        else:
            algo = ESPPRC(1, red_dem, red_tws, red_sts, N, red_tms, red_prices)
            opt_labels = algo.solve()
            print(len(opt_labels))
            if len(opt_labels) > 0:
                ordered_route = opt_labels[0].path()
                reduced_cost = opt_labels[0].cost
            else:
                ordered_route = []
                reduced_cost = 0

        ordered_route = remap_route(ordered_route, cus_mapping)

        # HERE

        print("The route with a reduced instance: " + str(ordered_route))
        print("with Objective: " + str(reduced_cost))
        # print("POMO Objective with ML Reduction: " + str(best_rewards[x].item()))
        # print("With optimal route: " + str([cus_mappings[x][j] for j in columns[x] if j != 0]))
        print("POMO Objective with no Reduction: " + str(best_rewards2[x].item()))
        print("With route: " + str([j for j in best_columns2[x]]))

        miscount = 0
        for node in best_columns2[x]:
            if node != 0 and node - 1 not in sorted_indices[x]:
                miscount += 1

        # print("The length of the benchmark route is: " + str(len(best_columns2[0]) - 2))
        # print("The remaining customers are: " + str(sorted_indices[x] + 1))
        print("The number of missing customers is: " + str(miscount))
        print("----------------")
        print("")
        Scores.append(miscount)
        # POMO_ML.append(best_rewards[x].item())
        POMO_Dual.append(best_rewards2[x].item())
        Exact.append(reduced_cost)

    print("On average, " + str(round(statistics.mean(Scores), 2)) + " nodes are missing. ")
    print("The mean exact score is: " + str(statistics.mean(Exact)))
    # print("where the mean POMO score with ML reduction is: " + str(statistics.mean(POMO_ML)))
    print("as opposed to its score with simple dual reduction: " + str(statistics.mean(POMO_Dual)))

    pickle_out = open('Reduction Results for N=' + str(num_customers) + " with reduction size " + str(reduction_size),
                      'wb')
    pickle.dump(Scores, pickle_out)
    pickle_out.close()


def main():
    random.seed(25)
    np.random.seed(25)
    torch.manual_seed(25)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=200)
    parser.add_argument('--data_size', type=int, default=20, help='No. of evaluation instances')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--reduction_size', type=int, default=40, help='Remaining Nodes in Graph')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs used.')
    parser.add_argument('--temperature', type=float, default=3.5, help='temperature for adj matrix')
    parser.add_argument('--nlayers', type=int, default=2, help='num of layers')
    parser.add_argument('--MIP', type=bool, default=False, help='use MIP for exact method.')
    parser.add_argument('--POMO_path', type=str, default='model50_max_t_data_Nby2', help='POMO model path')
    parser.add_argument('--POMO_epoch', type=int, default=200, help='POMO model epoch')

    args = parser.parse_args()
    num_customers = args.num_customers
    device = args.device
    LENGDATA = args.data_size
    MIP = args.MIP

    print("Instances are of size " + str(args.num_customers))

    model = GNN(input_dim=3, hidden_dim=args.hidden, output_dim=2, n_layers=args.nlayers)
    model_name = 'Saved_Models/PP_%d/scatgnn_layer_%d_hid_%d_model_%d_temp_3.500.pth' % \
                 (args.num_customers, args.nlayers, args.hidden, args.epochs)
    checkpoint_main = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint_main)

    POMO_params = {
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

    env_params = {'problem_size': num_customers,
                  'pomo_size': num_customers}

    POMO = Model(**POMO_params)
    device = torch.device('cpu')
    checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    POMO.load_state_dict(checkpoint['model_state_dict'])

    evaluate_reduction(LENGDATA, num_customers, model, args.temperature, args.reduction_size, MIP, env_params,
                       POMO)


if __name__ == "__main__":
    main()
