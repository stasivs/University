import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from dataset_B import all_B_set
from dataset_P import all_P_set
from dataset_E import all_E_set

class GeneticForCVRP:
    def __init__(
        self,
        population_size=450,
        generations=800,
        elite_size=250,
        mutation_rate=0.35,
        seed=42
    ):
        """
        Args:
            population_size (int): How many solutions in each generation.
            generations (int): Number of generations to run.
            elite_size (int): Number of top solutions to keep for the next generation.
            mutation_rate (float): Probability of mutation for each solution (Random permutation).
            seed (int): Random seed for reproducibility.
        """
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def euclidean_distance(city1, city2):
        """
        Calculate the distance between two points (x, y).
        """
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

    def create_individual(self, coordinates):
        """
        Create one individual by shuffling the list of cities (except the depo).
        """
        cities = list(range(2, len(coordinates) + 1))
        random.shuffle(cities)
        return cities

    def create_population(self, coordinates):
        """
        Create the initial population
        """
        return [self.create_individual(coordinates) for _ in range(self.population_size)]

    def calculate_cost(self, individual, demands, coordinates, capacity):
        """
        Calculate the cost of a solution and the routes.
        Args:
            individual (list[int]): The order of cities to visit.
            demands (list[int]): Demands of each city.
            coordinates (list[tuple]): Coordinates of cities and depot.
            capacity (int): Truck capacity.
        Returns:
            total_cost (float): Total route cost.
            routes (list[list[int]]): List of routes, each route is a list of cities.
        """
        routes = []
        route = []
        total_cost = 0
        current_load = 0

        for city in individual:
            city_demand = demands[city - 1]
            if current_load + city_demand <= capacity:
                route.append(city)
                current_load += city_demand
            else:
                routes.append(route)
                route = [city]
                current_load = city_demand

        if route:
            routes.append(route)

        for r in routes:
            if not r:
                continue
            route_cost = self.euclidean_distance(coordinates[0], coordinates[r[0] - 1])
            for i in range(len(r) - 1):
                route_cost += self.euclidean_distance(
                    coordinates[r[i] - 1],
                    coordinates[r[i + 1] - 1]
                )
            route_cost += self.euclidean_distance(coordinates[r[-1] - 1], coordinates[0])
            total_cost += route_cost

        return total_cost, routes

    def select_parents(self, population, demands, coordinates, capacity):
        """
        Select the top elite_size solutions based on their cost.
        """
        ranked = sorted(
            population,
            key=lambda x: self.calculate_cost(x, demands, coordinates, capacity)[0]
        )
        return ranked[: self.elite_size]

    def crossover(self, parent1, parent2):
        """
        Perform crossover.
        """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size

        child[start:end] = parent1[start:end]

        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene
        return child

    def mutate(self, individual):
        """
        Perform mutation: Permutation of random 2 genes
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def solve(self, demands, coordinates, capacity):
        """
        Run the genetic algorithm and return the best solution found.
        Returns:
            best_cost (float): Cost of the best solution.
            best_routes (list[list[int]]): Routes of the best solution.
        """
        population = self.create_population(coordinates)
        best_cost = float('inf')
        best_solution = None
        best_routes = []

        for _ in range(self.generations):
            population.sort(key=lambda x: self.calculate_cost(x, demands, coordinates, capacity)[0])
            current_best_cost, current_best_routes = self.calculate_cost(population[0], demands, coordinates, capacity)
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_solution = population[0]
                best_routes = current_best_routes

            next_generation = self.select_parents(population, demands, coordinates, capacity)

            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(next_generation, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)

            population = next_generation

        return best_cost, best_routes
    

def show_solution(best_known, best_cost, best_routes, demands, coordinates, capacity, solver):
    print("\n===== Локальный просмотр решения =====")
    print(f"Итоговая стоимость маршрута (Best Cost): {best_cost:.2f}")
    for i, route in enumerate(best_routes, start=1):
        route_cost, _ = solver.calculate_cost(route, demands, coordinates, capacity)
        print(f"Маршрут #{i}: {route}\n\tДлина (подмаршрута) = {route_cost:.2f}")

    plt.figure(figsize=(8, 6))
    for i, route in enumerate(best_routes, start=1):
        route_coords = [coordinates[0]] + [coordinates[c - 1] for c in route] + [coordinates[0]]
        x_vals, y_vals = zip(*route_coords)
        plt.plot(x_vals, y_vals, marker='o', label=f"Route #{i}")

    deviation = (best_cost - best_known) / best_known * 100.0
    plt.scatter(*zip(*coordinates), color='red', s=25, label='Cities')
    plt.title(f"VRP Routes\nTotal cost: {best_cost:.2f}. Best known: {best_known:.2f}. Dev: {deviation:.2f}")
    plt.legend()
    plt.grid()
    plt.show()


def run_on_set(tests, set_name, solver, results_list):
    """
    Run the solver on a given set of tests and store results in a list.
    Args:
        tests (list): List of test cases.
        set_name (str): The name of the test set (e.g., "B", "E", "P").
        solver (GeneticForCVRP): The genetic algorithm solver.
        results_list (list): List to store results for all tests.
    """
    deviations = []

    print(f"\n========== Running set {set_name} ==========")
    for idx, test_data in enumerate(tests, start=1):
        coordinates, demands, capacity, car, best_known = test_data
        start_time = time.perf_counter()
        best_cost, best_routes = solver.solve(demands, coordinates, capacity)
        end_time = time.perf_counter()

        deviation_percent = (best_cost - best_known) / best_known * 100.0

        results_list.append({
            "Set": set_name,
            "TestID": idx,
            "Time": round(end_time - start_time, 2),
            "BestFound": round(best_cost, 2),
            "BestKnown": best_known,
            "Deviation(%)": round(deviation_percent, 2)
        })

        deviations.append(deviation_percent)

        print(f"Test #{idx}: Found = {round(best_cost, 2)}, Known = {round(best_known, 2)}, Dev = {round(deviation_percent, 2)}%, Time = {round(end_time - start_time, 2)}s")

    if deviations:
        avg_dev = sum(deviations) / len(deviations)
        print(f"\nMean deviation on set {set_name}: {round(avg_dev, 2)}%\n")


def run_test(test_data, set_name, solver):
    """
    Run a single test case using the genetic solver.
    Returns the result as a dictionary.
    """
    coordinates, demands, capacity, car, best_known = test_data
    start_time = time.perf_counter()
    best_cost, best_routes = solver.solve(demands, coordinates, capacity)
    end_time = time.perf_counter()

    deviation_percent = (best_cost - best_known) / best_known * 100.0

    return {
        "Set": set_name,
        "TestID": None,  # Will be updated by the parent function
        "Time": round(end_time - start_time, 2),
        "BestFound": round(best_cost, 2),
        "BestKnown": best_known,
        "Deviation(%)": round(np.abs(deviation_percent), 2)
    }


def run_on_set_parallel(tests, set_name, solver, results_list):
    print(f"\n========== Running set {set_name} in parallel ==========")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_test, test_data, set_name, solver) for test_data in tests]

        deviations = []

        for idx, future in enumerate(futures, start=1):
            result = future.result()
            result["TestID"] = idx
            results_list.append(result)

            deviations.append(result['Deviation(%)'])
            print(f"Test #{idx}: Found = {result['BestFound']:.2f}, Known = {result['BestKnown']:.2f}, "
                  f"Dev = {result['Deviation(%)']:.2f}%, Time = {result['Time']:.2f}s")
            
        if deviations:
            avg_dev = sum(deviations) / len(deviations)
            print(f"\nMean deviation on set {set_name}: {round(avg_dev, 2)}%\n")


if __name__ == "__main__":
    solver = GeneticForCVRP(
        population_size=800,
        generations=1400,
        elite_size=300,
        mutation_rate=0.3,
        seed=42
    )

    results = []

    run_on_set_parallel(all_B_set(), "B", solver, results)
    run_on_set_parallel(all_E_set(), "E", solver, results)
    run_on_set_parallel(all_P_set(), "P", solver, results)


    df = pd.DataFrame(results, columns=["Set", "TestID", "Time", "BestFound", "BestKnown", "Deviation(%)"])
    df.to_csv("results.csv", index=False)