import numpy as np
import torch
import math
import random


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def remap_route(route, cus_mapping):
    for x in range(1, len(route) - 1):
        route[x] = cus_mapping[route[x]]
    return route


def retain_indices(output, reduction_size, policy="greedy", excludes=None):
    if policy == "greedy":
        indices = output.argsort(dim=1, descending=True)[:, :reduction_size, 0]
    elif policy == "random":
        indices = torch.zeros((0, reduction_size))
        for x in range(len(output)):
            exclude_indices_tensor = torch.tensor(excludes[x], dtype=torch.long)
            all_indices = torch.arange(len(output[x]))
            mask = torch.ones(len(output[x]), dtype=torch.bool)
            mask[exclude_indices_tensor] = False
            remaining_indices = all_indices[mask]
            random_ids = torch.randperm(len(remaining_indices))[:reduction_size]
            chosen = remaining_indices[random_ids]
            chosen = torch.reshape(chosen, (1, reduction_size))
            indices = torch.cat((indices, chosen), dim=0)
    else:
        indices = torch.zeros((0, reduction_size))
        for x in range(len(output)):
            indis = exclude_and_random_select_with_probs(excludes[x], reduction_size, output[x, :, 0])
            indis = torch.reshape(indis, (1, reduction_size))
            indices = torch.cat((indices, indis), dim=0)
    return indices


def exclude_and_random_select_with_probs(exclude_indices, subset_size, probs):
    # Create a tensor from the exclude_indices list
    exclude_indices_tensor = torch.tensor(exclude_indices, dtype=torch.long)

    # Get all possible indices of the tensor
    all_indices = torch.arange(len(probs))

    # Create a mask to exclude the specified indices
    mask = torch.ones(len(probs), dtype=torch.bool)
    mask[exclude_indices_tensor] = False

    # Select the remaining indices after exclusion
    remaining_indices = all_indices[mask]

    # Adjust the probability distribution to the remaining indices
    adjusted_probs = probs[mask]

    # Normalize the probabilities for the remaining indices
    adjusted_probs = adjusted_probs / adjusted_probs.sum()

    # Randomly select a subset of indices from the remaining indices based on the probability distribution
    selected_indices = torch.multinomial(adjusted_probs, subset_size, replacement=False)
    selected_indices = remaining_indices[selected_indices]

    return selected_indices


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


import numpy as np


def calculate_compatibility(time_windows, travel_times, service_times):
    n = len(travel_times)
    earliest = torch.reshape(time_windows[:, 0], (n, 1)) + torch.reshape(service_times, (n, 1)) \
               + travel_times
    feasibles = earliest - time_windows[:, 1]
    earliest[feasibles > 0] = math.inf
    latest = torch.reshape(time_windows[:, 1], (n, 1)) + torch.reshape(service_times, (n, 1)) \
             + travel_times
    latest = torch.minimum(latest, torch.reshape(time_windows[:, 1], (1, n)))
    latest[earliest == math.inf] = math.inf

    TC_early = torch.maximum(earliest, torch.reshape(time_windows[:, 0], (1, n)))
    TC_late = torch.maximum(latest, torch.reshape(time_windows[:, 0], (1, n)))
    TC_early.fill_diagonal_(math.inf)
    TC_late.fill_diagonal_(math.inf)

    return TC_early, TC_late


def BE2(prices, alpha):
    edge_count = math.ceil(alpha * len(prices) ** 2)
    flattened_tensor = prices.flatten()

    # Find the indices of the N lowest values
    n_lowest_indices = torch.topk(flattened_tensor, edge_count, largest=False).indices

    # Create a mask to set the rest of the values to 0
    mask = torch.zeros_like(flattened_tensor, dtype=torch.bool)
    mask[n_lowest_indices] = True

    # Set the values that are not among the N lowest to 0
    flattened_tensor[~mask] = 0

    # Reshape the tensor back to its original 2D shape
    red_prices = flattened_tensor.reshape(prices.shape)

    return red_prices


def reshape_problem(coords, demands, time_windows, duals, service_times, time_matrix, prices, red_dim):
    coords = torch.clone(coords)
    demands = torch.clone(demands)
    time_windows = torch.clone(time_windows)
    duals = torch.clone(duals)
    service_times = torch.clone(service_times)
    time_matrix = torch.clone(time_matrix)
    prices = torch.clone(prices)

    batch_size = len(coords)
    red_cor, red_tw = torch.zeros((batch_size, red_dim, 2)), torch.zeros((batch_size, red_dim, 2))
    red_dems, red_duls, red_sts = torch.zeros((batch_size, red_dim)), torch.zeros((batch_size, red_dim)), torch.zeros(
        (batch_size, red_dim))
    red_tts, red_prices = torch.zeros((batch_size, red_dim, red_dim)), torch.zeros((batch_size, red_dim, red_dim))
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

    # print("The problem has been reduced to size: " + str(len(red_cor[0]) - 1))
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

    # print("The problem has been reduced to size: " + str(len(coords) - 1))
    return coords, demands, time_windows, service_times, time_matrix, prices, cus_mapping


class ESPRCTW_RL_solver(object):
    def __init__(self, env, model, node_dist=None, indices=None, duals=None, cus_mapping=None,
                 tw_width=None, demands=None):
        self.env = env
        self.model = model
        self.node_dist = node_dist
        self.indices = indices
        self.duals = duals
        self.cus_mapping = cus_mapping
        self.demands = demands
        self.tw_width = tw_width

    def get_loss(self, sol_iter=1, baseline_score=0, dem_pen=0.15, tw_pen=0.075):
        supreme_rewards = torch.zeros(self.env.batch_size)
        supreme_columns = {}
        for x in range(self.env.batch_size):
            supreme_columns[x] = []

        for n in range(sol_iter):
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

            real_rewards = reward * -1
            best_rewards_indexes = real_rewards.argmin(dim=1)
            best_rewards = torch.diagonal(real_rewards[:, best_rewards_indexes], 0)
            best_columns = {}
            for x in range(self.env.batch_size):
                best_columns[x] = torch.tensor(decisions[:, x, best_rewards_indexes[x]], dtype=torch.int).tolist()
                if best_rewards[x] < supreme_rewards[x]:
                    supreme_rewards[x] = best_rewards[x]
                    supreme_columns[x] = best_columns[x]

        losses = 0
        penalties = 0
        if self.node_dist is not None:
            dims = self.node_dist.shape
            losses = torch.zeros(dims[0])
            penalties = torch.zeros(dims[0])
            for x in range(dims[0]):
                factor = supreme_rewards[x] - baseline_score[x]
                for i in range(dims[1]):
                    # if i in self.indices[x]:
                    # losses[x] -= self.node_dist[x, i, 0] * self.duals[x, i + 1]
                    # losses[x] += self.node_dist[x, i, 0] * max(
                    # dem_pen * self.demands[x, i + 1] - tw_pen * self.tw_width[x, i + 1], 0)
                    if i in self.indices[x]:
                        reduced_i = list(self.cus_mapping[x].keys())[list(self.cus_mapping[x].values()).index(i + 1)]
                        if reduced_i in supreme_columns[x]:
                            if factor < 0:
                                losses[x] += self.node_dist[x, i, 0] * -math.exp(self.duals[x, i + 1] + abs(factor))
                    # else:
                    # losses[x] += self.node_dist[x, i, 1] * self.duals[x, i + 1]
                    penalties[x] += self.node_dist[x, i, 0]
            penalties = penalties - self.env.problem_size
            penalties = torch.maximum(penalties, torch.zeros(penalties.shape))

        # SEE code_blocks.py
        return losses, supreme_columns, supreme_rewards, penalties


class Node_Reduction(object):
    def __init__(self, indices, coords):
        self.indices = indices
        self.coords = torch.clone(coords)

    def reduce_instance(self):
        for x in range(len(self.coords)):
            for y in range(1, len(self.coords[0])):
                if y - 1 not in self.indices[x]:
                    self.coords[x, y, :] = math.inf

        return self.coords

    def dual_based_elimination(self, duals):
        N = len(self.coords)
        assert N == len(duals)
        for x in range(1, N):
            if duals[x] == 0:
                self.coords[x, :] = math.inf
        return self.coords


def main():
    tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    exclude_indices = [1, 3, 5, 7]
    subset_size = 3

    # Example probability distribution for all elements
    probs = torch.tensor([0.1, 0.2, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1])

    selected_indices = exclude_and_random_select_with_probs(exclude_indices, subset_size, probs)

    print("Selected indices:", selected_indices)
    print("Selected elements:", tensor[selected_indices])


if __name__ == "__main__":
    main()
