import numpy as np
from geopy.distance import great_circle
from sklearn.metrics import pairwise_distances
import random
import pandas as pd
import math

# customer data: (latitude, longitude, demand)
customers = [
    (4.3555, 113.9777, 5),
    (4.3976, 114.0049, 8),
    (4.3163, 114.0764, 3),
    (4.3184, 113.9932, 6),
    (4.4024, 113.9896, 5),
    (4.4142, 114.0127, 8),
    (4.4804, 114.0734, 3),
    (4.3818, 114.2034, 6),
    (4.4935, 114.1828, 5),
    (4.4932, 114.1322, 8)
]

# extract locations and demands (aka capacities)
locations = [(c[0], c[1]) for c in customers]
capacities = [c[2] for c in customers]

# define capacity constraint per cluster
cluster_capacity = 30  # because the maximum capacity of vehicles is 30


# function to calculate the total demand of a cluster
def cluster_demand(cluster, capacities):
    return sum(capacities[i] for i in cluster)


# CLUSTER FIRST, ROUTE SECOND: CLUSTER FIRST STEP
# function to initialize K-medoids
def initialize_medoids(K, locations):
    indices = np.random.choice(len(locations), K, replace=False)
    return indices


# function to assign points to the nearest medoid with capacity constraint
def assign_points_to_medoids(medoids, locations, capacities, cluster_capacity):
    distances = pairwise_distances([locations[i] for i in medoids], locations,
                                   metric=lambda x, y: great_circle(x, y).km)
    clusters = np.full(len(locations), -1)
    cluster_loads = np.zeros(len(medoids))

    for i in np.argsort(distances.min(axis=0)):
        closest_medoid = np.argmin(distances[:, i])
        if cluster_loads[closest_medoid] + capacities[i] <= cluster_capacity:
            clusters[i] = closest_medoid
            cluster_loads[closest_medoid] += capacities[i]
        else:
            # find an alternative medoid
            for medoid in np.argsort(distances[:, i]):
                if cluster_loads[medoid] + capacities[i] <= cluster_capacity:
                    clusters[i] = medoid
                    cluster_loads[medoid] += capacities[i]
                    break

    return clusters


# function to update medoids
def update_medoids(clusters, K, locations):
    new_medoids = []
    for k in range(K):
        cluster_points = [i for i in range(len(clusters)) if clusters[i] == k]
        if cluster_points:
            distances = pairwise_distances([locations[i] for i in cluster_points],
                                           metric=lambda x, y: great_circle(x, y).km)
            medoid_index = cluster_points[np.argmin(distances.sum(axis=0))]
            new_medoids.append(medoid_index)
    return new_medoids


# main capacitated K-medoids algorithm
def capacitated_k_medoids(K, locations, capacities, cluster_capacity, max_iter=100):
    medoids = initialize_medoids(K, locations)
    for _ in range(max_iter):
        clusters = assign_points_to_medoids(medoids, locations, capacities, cluster_capacity)
        new_medoids = update_medoids(clusters, K, locations)
        if np.array_equal(medoids, new_medoids):
            break
        medoids = new_medoids
    return clusters, medoids


# set number of clusters
K = 3

# run the algorithm
clusters, medoids = capacitated_k_medoids(K, locations, capacities, cluster_capacity)

cluster_dict = {"Cluster": [], "Members": [], "Total Demands": [], "Vehicle Type": [], "Route": [], "Round Trip Distance": [], "Cost": [], "Distance Per Stop": []}

for k in range(K):
    cluster_members = [i + 1 for i in range(len(clusters)) if clusters[i] == k]
    total_capacity = cluster_demand([i for i in range(len(clusters)) if clusters[i] == k], capacities)
    cluster_dict["Cluster"].append(k + 1)
    cluster_dict["Members"].append(cluster_members)
    cluster_dict["Total Demands"].append(total_capacity)

    if total_capacity <= 25:
        cluster_dict["Vehicle Type"].append("Type A")
    else:
        cluster_dict["Vehicle Type"].append("Type B")



# CLUSTER FIRST, ROUTE SECOND: ROUTE SECOND STEP
class GeneticAlgorithmTSP:
    def __init__(self, depot, customers, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500):
        self.depot = depot
        self.customers = [depot] + customers  # Include depot as the first customer
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.num_customers = len(self.customers)
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.num_customers, self.num_customers))
        for i in range(self.num_customers):
            for j in range(self.num_customers):
                if i != j:
                    lat1, lon1 = self.customers[i]
                    lat2, lon2 = self.customers[j]
                    distance_matrix[i, j] = 100 * np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)
        return distance_matrix

    def _create_route(self):
        # Create a route that does not include the depot
        route = random.sample(range(1, self.num_customers), self.num_customers - 1)
        # Always start and end at the depot (index 0)
        return [0] + route + [0]

    def _initial_population(self):
        return [self._create_route() for _ in range(self.pop_size)]

    def _route_distance(self, route):
        return sum(self.distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

    def _rank_routes(self, population):
        fitness_results = {i: self._route_distance(population[i]) for i in range(len(population))}
        return sorted(fitness_results.items(), key=lambda x: x[1])

    def _selection(self, pop_ranked):
        selection_results = []
        df = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for _ in range(self.elite_size):
            selection_results.append(pop_ranked[_][0])
        for _ in range(len(pop_ranked) - self.elite_size):
            pick = 100 * random.random()
            for i in range(len(pop_ranked)):
                if pick <= df.iat[i, 3]:
                    selection_results.append(pop_ranked[i][0])
                    break
        return selection_results

    def _mating_pool(self, population, selection_results):
        return [population[i] for i in selection_results]

    def _breed(self, parent1, parent2):
        gene_a = int(random.random() * (len(parent1) - 2)) + 1
        gene_b = int(random.random() * (len(parent1) - 2)) + 1

        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)

        child_p1 = parent1[start_gene:end_gene]
        child_p2 = [item for item in parent2 if item not in child_p1 and item != 0]

        return [0] + child_p1 + child_p2 + [0]

    def _breed_population(self, matingpool):
        children = []
        length = len(matingpool) - self.elite_size
        pool = random.sample(matingpool, len(matingpool))

        for i in range(self.elite_size):
            children.append(matingpool[i])

        for i in range(length):
            child = self._breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children

    def _mutate(self, individual):
        for swapped in range(1, len(individual) - 1):
            if random.random() < self.mutation_rate:
                swap_with = int(random.random() * (len(individual) - 2)) + 1

                city1 = individual[swapped]
                city2 = individual[swap_with]

                individual[swapped] = city2
                individual[swap_with] = city1
        return individual

    def _mutate_population(self, population):
        return [self._mutate(ind) for ind in population]

    def _next_generation(self, current_gen):
        pop_ranked = self._rank_routes(current_gen)
        selection_results = self._selection(pop_ranked)
        matingpool = self._mating_pool(current_gen, selection_results)
        children = self._breed_population(matingpool)
        next_gen = self._mutate_population(children)
        return next_gen

    def solve(self):
        pop = self._initial_population()

        for _ in range(self.generations):
            pop = self._next_generation(pop)

        final_distance = str(self._rank_routes(pop)[0][1])
        best_route_index = self._rank_routes(pop)[0][0]
        best_route = pop[best_route_index]
        best_route_coordinates = [self.customers[i] for i in best_route]

        return best_route, best_route_coordinates, final_distance


depot = (4.4184, 114.0932)
best_route_list = []

for members in cluster_dict["Members"]:
    customers_location = []
    for member in members:
        customers_location.append(customers[member-1][:2])

    ga_tsp = GeneticAlgorithmTSP(depot, customers_location)
    best_route, best_route_coordinates, round_trip_distance = ga_tsp.solve()
    best_route_list.append(best_route)
    cluster_dict["Round Trip Distance"].append(float(round_trip_distance))


# COMPLETE 'cluster_dist'
# complete 'cluster_dist["Route"]'
def map_result_to_cluster_members(route_list, members):
    return [[0] + [members[idx - 1] for idx in route[1:-1]] + [0] for route in route_list]


# mapping results to cluster members
for i, members in enumerate(cluster_dict["Members"]):
    cluster_dict["Route"].append(map_result_to_cluster_members([best_route_list[i]], members)[0])


# complete 'cluster_dist["Cost"]'
# convert Round Trip Distance values to float for calculations
round_trip_distances = [float(dist) for dist in cluster_dict["Round Trip Distance"]]

# calculate cost based on vehicle type and append to the Cost list
for vehicle_type, distance in zip(cluster_dict["Vehicle Type"], round_trip_distances):
    if vehicle_type == 'Type A':
        cost = distance * 1.2
    elif vehicle_type == 'Type B':
        cost = distance * 1.5
    cluster_dict["Cost"].append(cost)


# complete 'cluster_dist["Distance Per Stop"]'
all_locations = [depot] + [(lat, lon) for lat, lon, demand in customers]


def euclidean_distance(coord1, coord2):
    return 100 * math.sqrt((coord2[1] - coord1[1]) ** 2 + (coord2[0] - coord1[0]) ** 2)


routes = cluster_dict['Route']

distances_per_route = []
for route in routes:
    distances_per_stop = []
    for i in range(len(route)-1):
        first = route[i]
        second = route[i+1]
        distance_between_stops = euclidean_distance(all_locations[first], all_locations[second])
        distances_per_stop.append(distance_between_stops)
    distances_per_route.append(distances_per_stop)

cluster_dict['Distance Per Stop'] = distances_per_route


# OUTPUT
print("\n\n\n\n\n----------OUTPUT----------")
total_distance = sum(cluster_dict["Round Trip Distance"])
total_distance = "{:.3f}".format(total_distance)
total_cost = sum(cluster_dict["Cost"])
total_cost = "{:.2f}".format(total_cost)

print(f"Total Distance = {total_distance}km")
print(f"Total Cost = RM{total_cost}")

for i in range(len(cluster_dict['Cluster'])):
    vehicle_num = cluster_dict['Cluster'][i]
    vehicle_type = cluster_dict['Vehicle Type'][i]
    round_trip_distance = cluster_dict['Round Trip Distance'][i]
    round_trip_distance = "{:.3f}".format(round_trip_distance)
    cost = cluster_dict['Cost'][i]
    cost = "{:.2f}".format(cost)
    demand = cluster_dict['Total Demands'][i]
    dist_stops = cluster_dict['Distance Per Stop'][i]

    for j, member in enumerate(cluster_dict['Route'][i][1:-1]):
        last = len(dist_stops)-1
        route = ' -> '.join(['Depot'] + [f"C{member}({dist_stops[j]:.3f}km)"] + [f"Depot({dist_stops[last]:.3f}km)"])

    print(f"Vehicle {vehicle_num} ({vehicle_type}):")
    print(f"Round Trip Distance: {round_trip_distance} km, Cost: RM{cost}, Demand: {demand}")
    print(f"{route}\n")
