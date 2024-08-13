from dataclasses import dataclass
import torch
import numpy

from ESPRCTWProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    time_windows: torch.Tensor = None
    duals: torch.Tensor = None
    service_times: torch.Tensor = None
    travel_times: torch.Tensor = None
    prices: torch.Tensor = None
    depot_time_window: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    current_times: torch.Tensor = None
    current_prices: torch.Tensor = None

class ESPRCTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        self.depot_node_service_time = None
        self.depot_node_time_windows = None
        self.depot_node_duals = None
        self.travel_times = None
        self.prices = None

        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        self.current_times = None
        self.current_prices = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def declare_problem(self, coords, demands, time_windows,
                        duals, service_times, travel_times, prices, batch_size):

        self.depot_node_xy = coords
        self.depot_node_demand = demands
        self.depot_node_duals = torch.tensor(duals, dtype=torch.float32)
        self.depot_node_time_windows = time_windows
        self.depot_node_service_time = service_times
        self.travel_times = travel_times
        self.prices = prices

        self.batch_size = batch_size

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        depot_xy = self.depot_node_xy[:, 0, :]
        depot_xy = depot_xy[:, None, :].expand(-1, 1, -1)
        depot_tw = self.depot_node_time_windows[:, 0, :]
        depot_tw = depot_tw[:, None, :].expand(-1, 1, -1)
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = self.depot_node_demand[:, 1:]
        self.reset_state.time_windows = self.depot_node_time_windows[:, 1:, :]
        self.reset_state.depot_time_window = depot_tw
        self.reset_state.travel_times = travel_times
        self.reset_state.prices = prices
        self.reset_state.duals = self.depot_node_duals[:, 1:]
        self.reset_state.service_times = self.depot_node_service_time[:, 1:]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.current_times = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.current_prices = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.current_times = self.current_times
        self.step_state.current_prices = self.current_prices
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)
        # Dynamic-1
        ####################################
        self.selected_count += 1

        if self.current_node is not None:
            previous_indexes = self.current_node
        else:
            previous_indexes = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.int64)

        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        if self.selected_count > 1:
            self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)

        service_time_list = self.depot_node_service_time[:, None, :].expand(self.batch_size, self.pomo_size, -1)

        travel_time_list = self.travel_times[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, problem+1)
        price_list = self.prices[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, problem+1)

        time_window_list = self.depot_node_time_windows[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        previous_indexes = previous_indexes[:, :, None]
        # shape: (batch, pomo)

        selected_service_times = service_time_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        previous_indexes = previous_indexes.expand(-1, -1, self.problem_size + 1)
        previous_indexes = previous_indexes[:, :, None, :]

        selected_travel_matrices = travel_time_list.gather(dim=2, index=previous_indexes).squeeze(dim=2)
        # shape: (batch, pomo, problem+1)
        selected_travel_times = selected_travel_matrices.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        selected_price_matrices = price_list.gather(dim=2, index=previous_indexes).squeeze(dim=2)
        # shape: (batch, pomo, problem+1)
        selected_prices = selected_price_matrices.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        gathering_index2 = selected[:, :, None].expand(-1, -1, 2)
        gathering_index2 = gathering_index2[:, :, None, :]

        selected_time_windows = time_window_list.gather(dim=2, index=gathering_index2).squeeze(dim=2)
        # shape: (batch, pomo, 2)

        gathering_index4 = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.int64)
        gathering_index4 = gathering_index4[:, :, None]
        selected_window_0 = selected_time_windows.gather(dim=2, index=gathering_index4).squeeze(dim=2)
        # shape: (batch, pomo)

        self.load -= selected_demand

        self.current_times = self.current_times + selected_travel_times
        self.current_times = torch.maximum(self.current_times, selected_window_0)
        self.current_times = self.current_times + selected_service_times

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)

        if self.selected_count == 2:
            self.visited_ninf_flag[:, :, 0] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list

        gathering_index5 = selected[:, :, None].expand(-1, -1, self.problem_size + 1)
        gathering_index5 = gathering_index5[:, :, None, :]
        future_travel_matrices = travel_time_list.gather(dim=2, index=gathering_index5).squeeze(dim=2)
        # shape: (batch, pomo, problem+1)

        gathering_index3 = torch.ones((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.int64)
        gathering_index3 = gathering_index3[:, :, :, None]
        future_window_1 = time_window_list.gather(dim=3, index=gathering_index3).squeeze(dim=3)
        # shape: (batch, pomo, problem+1)

        future_arrivals = self.current_times[:, :, None] + future_travel_matrices

        too_late = future_arrivals + round_error_epsilon > future_window_1

        gathering_index6 = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.int64)
        gathering_index6 = gathering_index6[:, :, :, None]
        future_window_0 = time_window_list.gather(dim=3, index=gathering_index6).squeeze(dim=3)
        # shape: (batch, pomo, problem+1)

        future_entries = torch.maximum(future_arrivals, future_window_0)
        future_departures = future_entries + service_time_list

        gathering_index7 = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.int64)
        gathering_index7 = gathering_index7[:, :, :, None]
        travel_time_to_depot = travel_time_list.gather(dim=3, index=gathering_index7).squeeze(dim=3)
        # shape: (batch, pomo, problem+1)

        depot_arrival = future_departures + travel_time_to_depot
        # shape: (batch, pomo, problem+1)
        depot_time_windows = self.depot_node_time_windows[:, 0, 1]
        depot_time_windows = depot_time_windows[:, None, None].expand(-1, self.pomo_size, self.problem_size + 1)

        cant_reach_depot = depot_arrival + round_error_epsilon > depot_time_windows
        # shape: (batch, pomo, problem+1)

        self.ninf_mask[demand_too_large] = float('-inf')
        self.ninf_mask[too_late] = float('-inf')
        self.ninf_mask[cant_reach_depot] = float('-inf')
        # shape: (batch, pomo, problem+1)

        self.current_prices = self.current_prices + selected_prices

        newly_finished = self.at_the_depot
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, :][self.finished] = float('-inf')
        self.ninf_mask[:, :, 0][self.finished] = 0

        # Consider updating state values only for episodes that have not terminated.
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.current_times = self.current_times
        self.step_state.current_prices = self.current_prices
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = self.current_prices
        else:
            reward = None

        return self.step_state, reward, done


def main():
    pass


if __name__ == "__main__":
    main()
