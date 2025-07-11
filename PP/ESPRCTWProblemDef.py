import math
import random

import torch
import numpy
from scipy.spatial import distance_matrix


def get_random_problems(batch_size, problem_size):
    coords = torch.rand(size=(batch_size, problem_size + 1, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20 or problem_size == 30 or problem_size == 10:
        demand_scaler = 30
    elif problem_size == 50 or problem_size == 40:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200 or problem_size == 500:
        demand_scaler = 50
    elif problem_size == 1000 or problem_size == 800:
        demand_scaler = 80
    else:
        raise NotImplementedError

    demands = torch.randint(1, 10, size=(batch_size, problem_size + 1)) / demand_scaler
    tw_scalar = 18
    lower_tw = torch.randint(0, 17, (batch_size, problem_size + 1))
    upper_tw = torch.randint(2, 9, (batch_size, problem_size + 1)) + lower_tw
    upper_tw = torch.minimum(upper_tw, torch.tensor([18]))
    time_windows = torch.zeros((batch_size, problem_size + 1, 2))
    time_windows[:, :, 0] = lower_tw
    time_windows[:, :, 1] = upper_tw
    time_windows = time_windows / tw_scalar
    service_times = (torch.rand((batch_size, problem_size + 1)) * 0.3 + 0.2) / tw_scalar
    travel_times = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    # prices = torch.zeros((batch_size, problem_size + 1, problem_size + 1))
    # duals = torch.zeros((batch_size, problem_size + 1))
    for x in range(batch_size):
        time_windows[x, 0] = torch.tensor([0, 1])
        service_times[x, 0] = 0
        demands[x, 0] = 0
        travel_times[x] = torch.FloatTensor(distance_matrix(coords[x], coords[x]))
        travel_times[x].fill_diagonal_(0)
        '''duals[x] = create_duals(1, problem_size, travel_times[x:x + 1])[0]
        prices[x] = travel_times[x] - duals[x]
        prices[x].fill_diagonal_(0)
        min_val = torch.min(prices[x])
        max_val = torch.max(prices[x])
        prices[x] = prices[x] / max(abs(max_val), abs(min_val))'''

    travel_times = travel_times / tw_scalar
    # duals = duals  # / tw_scalar

    print("Dataset created")
    return coords, demands, time_windows, service_times, travel_times


def create_duals(batch_size, problem_size, time_matrix):
    duals = torch.zeros(size=(batch_size, problem_size + 1), dtype=torch.float32)
    scaler = 0.2 + 0.9 * numpy.random.random()
    for x in range(batch_size):
        non_zeros = numpy.random.randint(problem_size / 2, problem_size + 1)
        indices = list(range(1, problem_size + 1))
        chosen = random.sample(indices, non_zeros)
        for index in chosen:
            duals[x, index] = torch.max(time_matrix[x, :, index]) * scaler * numpy.random.random()

    return duals
