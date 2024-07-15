from utils import *
from threading import Thread


class Subproblem:
    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows,
                 service_times, price, forbidden_edges):
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows

        self.service_times = service_times
        self.forbidden_edges = forbidden_edges

        self.price = price*-1

        self.primal_bound = 0
        self.primal_label = []

        self.price_arrangement = self.arrange_per_price()

    def arrange_per_price(self):
        arrangements = {}
        for cus in range(self.num_customers + 1):
            arrangements[cus] = np.argsort(self.price[cus, :])
        return arrangements

    def determine_PULSE_bounds(self, increment, stopping_time):
        print("Computing bounds")
        self.increment = increment
        self.no_of_increments = math.ceil(self.time_windows[0, 1] / self.increment - 1)
        self.bounds = np.zeros((self.num_customers, self.no_of_increments)) + math.inf
        self.supreme_labels = {}
        stopping_inc = math.ceil(stopping_time / self.increment - 1)

        for inc in range(self.no_of_increments, stopping_inc, -1):
            threads = []
            for cus in self.price_arrangement[0]:
                if cus == 0:
                    continue
                start_point = cus
                current_label = [cus]
                remaining_capacity = self.vehicle_capacity - self.demands[cus]
                current_time = self.time_windows[0, 1] - (self.no_of_increments - inc + 1) * increment
                current_price = 0
                best_bound = min(np.min(self.bounds[cus - 1, :]), 0)
                solve = False
                thread = Bound_Threader(target=self.bound_calculator, args=(start_point, current_label,
                                                                            remaining_capacity,
                                                                            current_time, current_price,
                                                                            best_bound, solve))
                thread.start()
                threads.append(thread)

            for index, thread in enumerate(threads):
                label, lower_bound = thread.join()
                self.bounds[index, inc - 1] = lower_bound
                self.supreme_labels[index + 1, inc] = label

    def bound_calculator(self, start_point, current_label, remaining_capacity, current_time,
                         current_price, best_bound, solve):

        if current_time > self.time_windows[start_point, 1] or remaining_capacity < 0:
            return [], math.inf

        if start_point == 0 and len(current_label) > 1:
            if current_price < -0.0001:
                if solve:
                    if current_price < self.primal_bound:
                        self.primal_bound = current_price
                        self.primal_label = current_label
            else:
                current_price = 0
            return current_label, current_price

        waiting_time = max(self.time_windows[start_point, 0] - current_time, 0)
        current_time += waiting_time
        current_time += self.service_times[start_point]

        inc = math.ceil(self.no_of_increments - (self.time_windows[0, 1] - current_time) / self.increment)
        if 0 < inc <= self.no_of_increments:
            if self.bounds[start_point - 1, inc - 1] < math.inf:
                bound_estimate = current_price + self.bounds[start_point - 1, inc - 1]
                if solve:
                    if bound_estimate > self.primal_bound:
                        return [], math.inf
                else:
                    if bound_estimate > best_bound:
                        return [], math.inf

        best_label = []
        best_price_indices = self.price_arrangement[start_point]
        for index in range(len(best_price_indices)):
            j = best_price_indices[index]
            if j > 0:
                if j in current_label:
                    continue
            else:
                if start_point == 0:
                    continue

            if [start_point, j] not in self.forbidden_edges and self.price[start_point, j] != math.inf:

                copy_label = current_label.copy()
                RC = remaining_capacity
                CT = current_time
                CP = current_price

                copy_label.append(j)
                RC -= self.demands[j]
                CT += self.time_matrix[start_point, j]
                CP += self.price[start_point, j]

                if len(copy_label) > 2 and j != 0:
                    roll_back_price = CP - (self.price[copy_label[-3], start_point] + self.price[start_point, j]) + \
                                      self.price[copy_label[-3], j]

                    roll_back_time = CT - (
                            self.time_matrix[start_point, j] + self.service_times[start_point] + waiting_time +
                            self.time_matrix[copy_label[-3], start_point])
                    roll_back_time += self.time_matrix[copy_label[-3], j]
                    roll_back_time = max(roll_back_time, self.time_windows[j, 0])

                    if roll_back_price <= CP and roll_back_time <= max(self.time_windows[j, 0], CT):
                        CT = math.inf

                label, lower_bound = self.bound_calculator(j, copy_label, RC, CT, CP,
                                                           best_bound, solve)

                if lower_bound < best_bound:
                    best_bound = lower_bound
                    best_label = label

        return best_label, best_bound

    def solve(self):

        self.determine_PULSE_bounds(self.time_windows[0, 1]/9, 0.5 * self.time_windows[0, 1])
        print("Bounds Computed")

        threads = []
        best_routes = []
        best_costs = []
        self.primal_bound = min(np.min(self.bounds), 0)
        for cus in self.price_arrangement[0]:
            start_point = cus
            if (0, cus) not in self.forbidden_edges:
                current_label = [0, cus]
                remaining_capacity = self.vehicle_capacity - self.demands[cus]
                current_time = self.time_matrix[0, cus]
                current_price = self.price[0, cus]
                best_bound = 0
                solve = True
                thread = Bound_Threader(target=self.bound_calculator, args=(start_point, current_label,
                                                                            remaining_capacity,
                                                                            current_time, current_price,
                                                                            best_bound, solve))
                thread.start()
                threads.append(thread)

        for index, thread in enumerate(threads):
            label, cost = thread.join()
            best_routes.append(label)
            best_costs.append(cost)

        best_cost = min(best_costs)
        best_index = best_costs.index(best_cost)
        best_route = best_routes[best_index]
        best_routes.remove(best_route)
        best_costs.remove(best_cost)
        promising_labels = [best_routes[x] for x in range(len(best_routes)) if best_costs[x] < -0.001]
        return best_route, best_cost, promising_labels


class Bound_Threader(Thread):

    def __init__(self, target, args):
        Thread.__init__(self, target=target, args=args)
        self._return = None
        self.daemon = True

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return
