import math
import random

import torch
import numpy


def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 100
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)
    depot_time_window = torch.tensor([0, 1]).repeat(batch_size, 1, 1)
    # shape: (batch, 1, 2)
    tw_scalar = 18
    time_windows = create_time_windows(batch_size, problem_size, tw_scalar) / float(tw_scalar)
    service_times = create_service_times(batch_size, problem_size) / float(tw_scalar)
    travel_times = create_time_matrix(batch_size, problem_size, node_xy, depot_xy)
    # duals = create_duals(batch_size, problem_size, tw_scalar)
    duals = create_duals_2(batch_size, problem_size, travel_times)
    prices = create_price(travel_times, duals)
    travel_times = travel_times / float(tw_scalar)
    duals = duals / float(tw_scalar)

    # duals = torch.tensor(duals / tw_scalar, dtype=torch.float32)
    print("Dataset created")
    return depot_xy, node_xy, node_demand, time_windows, depot_time_window, duals, service_times, travel_times, prices


def create_service_times(batch_size, problem_size):
    service_times = torch.zeros(size=(batch_size, problem_size), dtype=torch.float32)
    for x in range(batch_size):
        service_times[x, :] = 0.3 * torch.rand(problem_size) + 0.2
    return service_times


def create_duals(batch_size, problem_size, tw_scaler):
    duals = torch.zeros(size=(batch_size, problem_size), dtype=torch.float32)
    for x in range(batch_size):
        inst_dist = numpy.random.random_sample()
        if inst_dist < 0.1:
            non_zeros = numpy.random.randint(4, math.ceil(problem_size / 4))
            upper_bound = 5
            dist = "uni"
        elif inst_dist < 0.35:
            non_zeros = numpy.random.randint(math.ceil(problem_size / 4), math.ceil(problem_size * 3 / 4))
            upper_bound = 1.5
            if non_zeros < 0.4 * problem_size:
                dist = "uni"
            else:
                dist = "exp"
        else:
            non_zeros = numpy.random.randint(math.ceil(problem_size * 3 / 4), problem_size + 1)
            upper_bound = 1
            dist = "exp"

        indices = list(range(problem_size))
        chosen = random.sample(indices, non_zeros)
        for index in chosen:
            if dist == "uni":
                duals[x, index] = upper_bound * numpy.random.random_sample()
            else:
                duals[x, index] = min(numpy.random.exponential(0.33), upper_bound)

    return duals


def create_duals_2(batch_size, problem_size, time_matrix):
    duals = torch.zeros(size=(batch_size, problem_size), dtype=torch.float32)
    for x in range(batch_size):
        scaler = 1.1  # 0.2 + 0.9 * numpy.random.random()
        non_zeros = numpy.random.randint(problem_size / 2, problem_size + 1)
        indices = list(range(problem_size))
        chosen = random.sample(indices, non_zeros)
        for index in chosen:
            duals[x, index] = torch.max(time_matrix[x, :, index + 1]) * scaler * numpy.random.random()

    return duals


def create_time_matrix(batch_size, problem_size, node_coors, depot_coors):
    time_matrix = torch.zeros(size=(batch_size, problem_size + 1, problem_size + 1), dtype=torch.float32)
    for x in range(batch_size):
        for i in range(problem_size + 1):
            for j in range(problem_size + 1):
                if i != j:
                    if i == 0:
                        time_matrix[x, i, j] = torch.linalg.norm(depot_coors[x, i, :] - node_coors[x, j - 1, :])
                    elif j == 0:
                        time_matrix[x, i, j] = torch.linalg.norm(node_coors[x, i - 1, :] - depot_coors[x, j, :])
                    else:
                        time_matrix[x, i, j] = torch.linalg.norm(node_coors[x, i - 1, :] - node_coors[x, j - 1, :])

    return time_matrix


def create_price(time_matrix, duals):
    batch_size, dim_1, dim_2 = time_matrix.shape
    prices = torch.zeros(size=(batch_size, dim_1, dim_2), dtype=torch.float32)
    for x in range(batch_size):
        for j in range(dim_1):
            if j != 0:
                prices[x, j, :] = (time_matrix[x, j, :] - duals[x, j - 1]) * -1
            else:
                prices[x, j, :] = time_matrix[x, j, :]
        min_val = torch.min(prices[x, :, :])
        max_val = torch.max(prices[x, :, :])
        prices[x, :, :] = prices[x, :, :] / max(abs(max_val), abs(min_val))
    return prices


def create_time_windows(batch_size, problem_size, tw_scalar, minimum_margin=2):
    time_windows = torch.zeros(size=(batch_size, problem_size, 2), dtype=torch.float32)
    for x in range(batch_size):
        for i in range(problem_size):
            time_windows[x, i, 0] = numpy.random.randint(0, 10)
            time_windows[x, i, 1] = numpy.random.randint(time_windows[x, i, 0] + minimum_margin, tw_scalar)
    return time_windows


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems
