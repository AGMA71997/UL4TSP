from ESPRCTWProblemDef import get_random_problems
from ESPRCTWEnv import ESPRCTWEnv as Env
from ESPRCTWModel import ESPRCTWModel as Model
from PP_exact_solver import Subproblem
from utils import *
from models import GNN

import random
import numpy as np
import argparse
import torch
import pickle
import statistics


def evaluate_reduction(LENGDATA, num_customers, model, temperature, reduction_size, MIP, env_params, POMO):
    depot_xy, coords, demands, time_windows, depot_time_window, duals, service_times, travel_times, prices = \
        get_random_problems(LENGDATA, num_customers)
    coords, time_windows, demands, service_times, duals = merge_with_depot(depot_xy, coords,
                                                                           demands, time_windows, depot_time_window,
                                                                           duals,
                                                                           service_times)

    f0 = coords[:, 1:, :]
    f1 = time_windows[:, 1:, 1] - time_windows[:, 1:, 0]
    f2 = duals[:, 1:]
    f3 = demands[:, 1:]
    dims = f2.shape
    f1 = torch.reshape(f1, (dims[0], dims[1], 1))
    f2 = torch.reshape(f2, (dims[0], dims[1], 1))
    f3 = torch.reshape(f3, (dims[0], dims[1], 1))
    X = torch.cat([f0, f1, f2, f3], dim=2)
    distance_m = travel_times[:, 1:, 1:]
    adj = torch.exp(-1. * distance_m / temperature)
    output = model(X, adj)
    probas = torch.sort(output, 1, descending=True)[0]
    print(probas[5, :, 0])

    '''exclus = {}
    for inst in range(len(coords)):
        exclus[inst] = [j - 1 for j in range(1, len(duals[inst])) if duals[inst, j] == 0]'''

    sorted_indices = retain_indices(output, reduction_size)

    NR = Node_Reduction(sorted_indices, coords)
    red_cor = NR.reduce_instance()
    red_cor, red_dem, red_tws, red_dualsy, red_sts, red_tms, red_prices, cus_mappings = reshape_problem(red_cor,
                                                                                                        demands,
                                                                                                        time_windows,
                                                                                                        duals,
                                                                                                        service_times,
                                                                                                        travel_times,
                                                                                                        prices,
                                                                                                        reduction_size + 1)
    env = Env(**env_params)
    env.declare_problem(red_cor, red_dem, red_tws, red_dualsy, red_sts, red_tms, red_prices, LENGDATA)
    pp_rl_solver = ESPRCTW_RL_solver(env, POMO)
    loss, columns, best_rewards, penalties = pp_rl_solver.get_loss(sol_iter=1)
    Scores = []
    print("Reduction Made")

    for x in range(LENGDATA):

        NR = Node_Reduction(None, coords[x])
        # red_cor = NR.dual_based_elimination(duals[x])
        red_cor = torch.clone(coords[x])
        red_cor, red_dem, red_tws, red_sts, red_tms, red_prices, cus_mapping = reshape_problem_2(red_cor,
                                                                                                 demands[x],
                                                                                                 time_windows[x],
                                                                                                 service_times[x],
                                                                                                 travel_times[x],
                                                                                                 prices[x])

        N = len(red_cor) - 1
        '''subproblem = Subproblem(N, 1, red_tms, red_dem, red_tws, red_sts, red_prices, [])

        if MIP:
            ordered_route, reduced_cost = subproblem.MIP_program()
        else:
            ordered_route, reduced_cost, top_labels = subproblem.solve()

        ordered_route = remap_route(ordered_route, cus_mapping)'''
        ordered_route = []
        reduced_cost = 0

        # HERE
        indices = torch.tensor([cus_mapping[j] - 1 for j in cus_mapping])
        # indices = torch.tensor([j - 1 for j in ordered_route if j != 0])
        indices = torch.reshape(indices, (1, len(indices)))
        NR = Node_Reduction(indices, coords[x:x + 1])
        red_cor = NR.reduce_instance()
        red_cor, red_dem, red_tws, red_duals, red_sts, red_tms, red_prices, cus_mapping2 = reshape_problem(red_cor,
                                                                                                          demands[
                                                                                                          x:x + 1],
                                                                                                          time_windows[
                                                                                                          x:x + 1],
                                                                                                          duals[
                                                                                                          x:x + 1],
                                                                                                          service_times[
                                                                                                          x:x + 1],
                                                                                                          travel_times[
                                                                                                          x:x + 1],
                                                                                                          prices[
                                                                                                          x:x + 1],
                                                                                                          N + 1)
        env_params_2 = {'problem_size': N,
                        'pomo_size': N}
        # print(torch.sort(red_dualsy[x], descending=True)[0])
        # print(torch.sort(red_duals, descending=True)[0])
        # print("**********")
        env = Env(**env_params_2)
        env.declare_problem(red_cor, red_dem, red_tws, red_duals, red_sts, red_tms, red_prices, 1)
        pp_rl_solver = ESPRCTW_RL_solver(env, POMO)
        loss, best_columns2, best_rewards2, penalties = pp_rl_solver.get_loss(sol_iter=1)

        # TO HERE

        print(ordered_route)
        print("Exact Objective: " + str(reduced_cost))
        print("POMO Objective with ML Reduction: " + str(best_rewards[x].item()))
        print("With optimal route: " + str([cus_mappings[x][j] for j in columns[x] if j != 0]))
        print("POMO Objective with no Reduction: " + str(best_rewards2[0].item()))
        print("With optimal route: " + str([cus_mapping2[0][j] for j in best_columns2[0] if j != 0]))

        miscount = 0
        for node in best_columns2[0]:
            if node != 0 and node - 1 not in sorted_indices[x]:
                miscount += 1

        print("The length of the benchmark route is: " + str(len(best_columns2[0]) - 2))
        print("The remaining customers are: " + str(sorted_indices[x] + 1))
        print("The number of missing customers is: " + str(miscount))
        print("----------------")
        print("")
        Scores.append(miscount / (len(best_columns2[0]) - 2))

    print("On average, " + str(round(statistics.mean(Scores), 2) * 100) + "% of nodes are missing. ")
    pickle_out = open('Reduction Results for N=' + str(num_customers) + " with reduction size " + str(reduction_size),
                      'wb')
    pickle.dump(Scores, pickle_out)
    pickle_out.close()


def main():
    random.seed(25)
    np.random.seed(25)
    torch.manual_seed(25)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=50)
    parser.add_argument('--data_size', type=int, default=20, help='No. of evaluation instances')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--reduction_size', type=int, default=20, help='Remaining Nodes in Graph')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs used.')
    parser.add_argument('--temperature', type=float, default=3.5, help='temperature for adj matrix')
    parser.add_argument('--nlayers', type=int, default=3, help='num of layers')
    parser.add_argument('--MIP', type=bool, default=True, help='use MIP for exact method.')
    parser.add_argument('--POMO_path', type=str, default='model20_max_t_data_Nby2', help='POMO model path')
    parser.add_argument('--POMO_epoch', type=int, default=200, help='POMO model epoch')

    args = parser.parse_args()
    num_customers = args.num_customers
    device = args.device
    LENGDATA = args.data_size
    MIP = args.MIP

    print("Instances are of size " + str(args.num_customers))

    model = GNN(input_dim=5, hidden_dim=args.hidden, output_dim=2, n_layers=args.nlayers)
    model_name = 'Saved_Models/PP_%d/scatgnn_layer_%d_hid_%d_model_%d_temp_3.500.pth' %\
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

    env_params = {'problem_size': args.reduction_size,
                  'pomo_size': args.reduction_size}

    POMO = Model(**POMO_params)
    device = torch.device('cpu')
    checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    POMO.load_state_dict(checkpoint['model_state_dict'])

    evaluate_reduction(LENGDATA, num_customers, model, args.temperature, args.reduction_size, MIP, env_params,
                       POMO)


if __name__ == "__main__":
    main()
