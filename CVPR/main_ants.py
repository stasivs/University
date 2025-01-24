import math
import random
import copy
import pandas as pd

from dataset_B import all_B_set
from dataset_E import all_E_set
from dataset_P import all_P_set

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class ACOForCVRP:
    def __init__(
        self,
        coordinates,
        demands,
        capacity,
        num_vehicles,
        ants_count=10,
        alpha=1.0,
        beta=5.0,
        evaporation_rate=0.5,
        iterations=200,
        seed=42
    ):
        """
        :param coordinates
        :param demands
        :param capacity
        :param num_vehicles
        :param ants_count
        :param alpha
        :param beta
        :param evaporation_rate
        :param iterations
        :param seed
        """
        random.seed(seed)
        self.coords = coordinates
        self.demands = demands
        self.capacity = capacity
        self.num_vehicles = num_vehicles

        self.n = len(coordinates)
        self.ants_count = ants_count
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.iterations = iterations

        self.dist_matrix = [[0.0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.dist_matrix[i][j] = euclidean_distance(coordinates[i], coordinates[j])
                else:
                    self.dist_matrix[i][j] = 999999

        self.pheromone = [[1.0]*(self.n) for _ in range(self.n)]
        
        self.best_route = None
        self.best_cost = float('inf')

    def run(self):
        """
        Run ants
        """
        for it in range(1, self.iterations+1):
            all_solutions = []
            for ant in range(self.ants_count):
                solution, total_cost = self.construct_solution()
                if solution is not None:
                    all_solutions.append((solution, total_cost))

            if not all_solutions:
                continue

            iteration_best = min(all_solutions, key=lambda x: x[1])
            if iteration_best[1] < self.best_cost:
                self.best_route = iteration_best[0]
                self.best_cost = iteration_best[1]

            self.update_pheromone(iteration_best)

        return self.best_route, self.best_cost

    def construct_solution(self):
        """
        Return solution per ant
        """
        unvisited = set(range(1, self.n))
        routes = []
        total_cost = 0.0
        vehicles_used = 0

        while unvisited and vehicles_used < self.num_vehicles:
            route_cost, route = self.build_one_route(unvisited)
            if route is None:
                return None, float('inf')
            vehicles_used += 1
            total_cost += route_cost
            routes.append(route)

        if unvisited:
            return None, float('inf')

        return routes, total_cost

    def build_one_route(self, unvisited):
        """
        Build route
        """
        current_capacity = self.capacity
        route = [0]
        cost = 0.0
        current_node = 0

        while True:
            possible_next = []
            for j in unvisited:
                if self.demands[j] <= current_capacity:
                    possible_next.append(j)

            if not possible_next:
                if current_node != 0:
                    cost += self.dist_matrix[current_node][0]
                    route.append(0)
                return cost, route

            next_node = self.select_next_node(current_node, possible_next)
            if next_node is None:
                if current_node != 0:
                    cost += self.dist_matrix[current_node][0]
                    route.append(0)
                return cost, route

            cost += self.dist_matrix[current_node][next_node]
            route.append(next_node)
            current_capacity -= self.demands[next_node]
            unvisited.remove(next_node)
            current_node = next_node

    def select_next_node(self, current_node, candidates):
        """
        Prob ~ (tau^alpha) * (eta^beta).
        """
        pheromone_values = []
        for j in candidates:
            tau = self.pheromone[current_node][j]
            eta = 1.0 / (self.dist_matrix[current_node][j] + 1e-9)
            pheromone_values.append((tau ** self.alpha) * (eta ** self.beta))

        s = sum(pheromone_values)
        if s <= 1e-9:
            return None
        
        r = random.random() * s
        cum = 0.0
        for idx, node_j in enumerate(candidates):
            cum += pheromone_values[idx]
            if cum >= r:
                return node_j
        return None

    def update_pheromone(self, iteration_best):
        """
        Remove phermone on p
        """
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.rho)
                if self.pheromone[i][j] < 1e-9:
                    self.pheromone[i][j] = 1e-9

        best_sol, best_cost = iteration_best
        if best_cost <= 0:
            return
        
        delta = 1.0 / best_cost
        for route in best_sol:
            for k in range(len(route) - 1):
                i = route[k]
                j = route[k+1]
                self.pheromone[i][j] += delta
                self.pheromone[j][i] += delta


def solve_problems(all_tests):
    results = []
    for idx, test_data in enumerate(all_tests, start=1):
        coordinates, demands, capacity, car, best_known = test_data
        
        aco = ACOForCVRP(
            coordinates=coordinates,
            demands=demands,
            capacity=capacity,
            num_vehicles=car,
            ants_count=30,
            alpha=0.7,
            beta=1.2,
            evaporation_rate=0.4,
            iterations=800,
            seed=42
        )

        best_route, best_cost = aco.run()

        deviation = (best_cost - best_known) / best_known * 100.0
    
        results.append((idx, best_cost, best_known, deviation))

    return results


def print_results(tests, name, verbose=False):
    final_results = solve_problems(tests)
    
    ans = []
    for r in final_results:
        test_id, found, known, dev = r

        ans.append(dev)
        
        if verbose:
            print(f"{name}-{test_id} -> Cost found: {found:.2f}, known: {known}, dev = {dev:.2f}%")

    print(f"Средняя погрешность на {name} сете: {round(sum(ans)/len(ans), 2)}")


if __name__ == "__main__":
    print("\n\nStarted processing B set")
    print_results(all_B_set(), "B", True)

    print("\n\nStarted processing E set")
    print_results(all_E_set(), "E", True)

    print("\n\nStarted processing P set")
    print_results(all_P_set(), "P", True)