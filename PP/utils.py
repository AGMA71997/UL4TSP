'''
adapted from Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks
'''
import numpy as np
import torch
from scipy.spatial import distance_matrix
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def coord_to_adj(coord_arr, duals):
    time_matrix = distance_matrix(coord_arr, coord_arr)
    prices = create_price(time_matrix, duals)
    return time_matrix, prices


def check_route_feasibility(route, time_matrix, time_windows, service_times, demands_data, truck_capacity):
    current_time = max(time_matrix[0, route[1]], time_windows[route[1], 0])
    total_capacity = 0

    for i in range(1, len(route)):
        if current_time > time_windows[route[i], 1]:
            print("Time Window violated")
            print(route[i])
            return False
        current_time += service_times[route[i]]
        total_capacity += demands_data[route[i]]
        if total_capacity > truck_capacity:
            print("Truck Capacity Violated")
            print(route[i])
            return False
        if i < len(route) - 1:
            # travel to next node
            current_time += time_matrix[route[i], route[i + 1]]
            current_time = max(current_time, time_windows[route[i + 1], 0])
    return True


def create_price(time_matrix, duals):
    if len(duals) < len(time_matrix):
        duals.insert(0, 0)
    assert duals[0] == 0 and len(duals) == len(time_matrix)
    duals = np.array(duals)
    prices = (time_matrix - duals) * -1
    np.fill_diagonal(prices, 0)
    return prices


def remap_route(route, cus_mapping):
    for x in range(1, len(route) - 1):
        route[x] = cus_mapping[route[x]]
    return route


def merge_with_depot(depot_xy, coords, node_demand, time_windows, depot_time_window,
                     duals, service_times):
    batch_size = len(coords)
    coords = torch.cat((depot_xy, coords), dim=1)
    time_windows = torch.cat((depot_time_window, time_windows), dim=1)
    depot_demand = torch.zeros(size=(batch_size, 1))
    demands = torch.cat((depot_demand, node_demand), dim=1)
    depot_st = torch.zeros(size=(batch_size, 1))
    service_times = torch.cat((depot_st, service_times), dim=1)
    depot_dual = torch.zeros(size=(batch_size, 1))
    duals = torch.cat((depot_dual, duals), dim=1)
    return coords, time_windows, demands, service_times, duals


def reshape_problem(coords, demands, time_windows, duals, service_times, time_matrix, prices, red_dim):
    coords = np.copy(coords)
    demands = np.copy(demands)
    time_windows = np.copy(time_windows)
    duals = np.copy(duals)
    service_times = np.copy(service_times)
    time_matrix = np.copy(time_matrix)
    prices = np.copy(prices)

    batch_size = len(coords)
    red_cor, red_tw = np.zeros((batch_size, red_dim, 2)), np.zeros((batch_size, red_dim, 2))
    red_dems, red_duls, red_sts = np.zeros((batch_size, red_dim)), np.zeros((batch_size, red_dim)), np.zeros(
        (batch_size, red_dim))
    red_tts, red_prices = np.zeros((batch_size, red_dim, red_dim)), np.zeros((batch_size, red_dim, red_dim))
    cus_mappings = []
    for x in range(len(coords)):
        remaining_customers = []
        cus_map = {}
        for y in range(1, len(coords[0])):
            if coords[x, y, 0] == math.inf:
                demands[x, y] = math.inf
                time_windows[x, y, :] = math.inf
                duals[x, y] = math.inf
                service_times[x, y] = math.inf
                time_matrix[x, y, :] = math.inf
                time_matrix[x, :, y] = math.inf
                prices[x, y, :] = math.inf
                prices[x, :, y] = math.inf
            else:
                remaining_customers.append(y)

        for y in range(len(remaining_customers)):
            cus_map[y + 1] = remaining_customers[y]

        cus_mappings.append(cus_map)

        red_cor[x] = coords[x, coords[x, :, 0] != math.inf]
        red_dems[x] = demands[x, demands[x, :] != math.inf]
        red_tw[x] = time_windows[x, time_windows[x, :, 0] != math.inf]
        red_duls[x] = duals[x, duals[x, :] != math.inf]
        red_sts[x] = service_times[x, service_times[x, :] != math.inf]
        tm = time_matrix[x, time_matrix[x, :, 0] != math.inf]
        mask = (tm == math.inf)
        idx = mask.any(axis=0)
        red_tts[x] = tm[:, ~idx]

        pm = prices[x, prices[x, :, 0] != math.inf]
        mask = (pm == math.inf)
        idx = mask.any(axis=0)
        red_prices[x] = pm[:, ~idx]

    print("The problem has been reduced to size: " + str(len(red_cor[0]) - 1))
    return red_cor, red_dems, red_tw, red_duls, red_sts, red_tts, red_prices, cus_mappings


def reshape_problem_2(coords, demands, time_windows, service_times, time_matrix, prices):
    coords = np.copy(coords)
    demands = np.copy(demands)
    time_windows = np.copy(time_windows)
    service_times = np.copy(service_times)
    time_matrix = np.copy(time_matrix)
    prices = np.copy(prices)

    remaining_customers = []
    for x in range(1, len(coords)):
        if coords[x, 0] == math.inf:
            demands[x] = math.inf
            time_windows[x, :] = math.inf
            service_times[x] = math.inf
            time_matrix[x, :] = math.inf
            time_matrix[:, x] = math.inf
            prices[x, :] = math.inf
            prices[:, x] = math.inf
        else:
            remaining_customers.append(x)

    cus_mapping = {}
    for x in range(len(remaining_customers)):
        cus_mapping[x + 1] = remaining_customers[x]

    coords = coords[coords[:, 0] != math.inf]
    demands = demands[demands[:] != math.inf]
    time_windows = time_windows[time_windows[:, 0] != math.inf]
    service_times = service_times[service_times[:] != math.inf]
    time_matrix = time_matrix[time_matrix[:, 0] != math.inf]
    mask = (time_matrix == math.inf)
    idx = mask.any(axis=0)
    time_matrix = time_matrix[:, ~idx]
    prices = prices[prices[:, 0] != math.inf]
    mask = (prices == math.inf)
    idx = mask.any(axis=0)
    prices = prices[:, ~idx]

    print("The problem has been reduced to size: " + str(len(coords) - 1))
    return coords, demands, time_windows, service_times, time_matrix, prices, cus_mapping


class ESPRCTW_RL_solver(object):
    def __init__(self, env, model, prices):
        self.env = env
        self.model = model
        self.prices = prices

    def train(self, steps):
        pass

    def evaluate(self):
        pass

    def return_real_reward(self, decisions):
        real_rewards = torch.zeros((self.env.batch_size, self.env.pomo_size))
        for x in range(self.env.batch_size):
            for y in range(self.env.pomo_size):
                real_rewards[x, y] = sum(
                    [self.prices[x, int(decisions[r, x, y]), int(decisions[r + 1, x, y])] for r in
                     range(len(decisions) - 1)])

        return real_rewards * -1

    def generate_columns(self):
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        decisions = torch.empty((0, self.env.batch_size, self.env.pomo_size), dtype=torch.float32)
        while not done:
            selected, _ = self.model(state)
            decisions = torch.cat((decisions, selected[None, :, :]), dim=0)
            # shape: (max episode length, batch, pomo)
            state, reward, done = self.env.step(selected)
            # shape: (batch, pomo)

        real_rewards = self.return_real_reward(decisions)
        real_rewards.requires_grad_(True)

        best_rewards_indexes = real_rewards.argmin(dim=1)

        best_columns = torch.tensor(decisions[:, :, best_rewards_indexes[0]], dtype=torch.int).tolist()

        promising_columns = {}
        for batch in range(self.env.batch_size):
            negative_reduced_costs = real_rewards[batch, :] < -0.0000001
            indices = negative_reduced_costs.nonzero()
            promising_columns[batch] = []
            for index in indices:
                column = torch.tensor(decisions[:, batch, index], dtype=torch.int)
                column = column.tolist()
                promising_columns[batch].append(column)

        return promising_columns, best_columns, torch.diagonal(real_rewards[:, best_rewards_indexes], 0)


class Node_Reduction(object):
    def __init__(self, indices, coords):
        self.indices = indices
        self.coords = np.copy(coords)

    def reduce_instance(self):
        for x in range(len(self.coords)):
            for y in range(1, len(self.coords[0])):
                if y - 1 not in self.indices[x, :]:
                    self.coords[x, y, :] = math.inf

        return self.coords

    def dual_based_elimination(self, duals):
        N = len(self.coords)
        for x in range(1, N):
            if duals[x] == 0:
                self.coords[x, :] = math.inf
        return self.coords
