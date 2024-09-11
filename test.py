from monte_carlo import *

# Create a random graph
num_nodes = 100  # Number of nodes
edge_density = 0.15  # Edge density
G = nx.gnp_random_graph(num_nodes, edge_density)  # Generates a random graph

# Set weight ranges for each edge (uniform distribution)
weight_ranges = {}  # {(u, v): (min_weight, max_weight)}
for (u, v) in G.edges():
    min_weight = random.uniform(1, 5)  # Generate minimum weight
    max_weight = random.uniform(min_weight, min_weight + 10) # Generate maximum weight
    weight_ranges[(u, v)] = (min_weight, max_weight) 
    weight_ranges[(v, u)] = (min_weight, max_weight)  # Ensure both directions are stored

# Define start and end nodes, and number of traversals (k) and simulations
start_node = 0 
end_node = num_nodes - 1
k = 10
simulations = 1000

# Initialize the MCTS graph traversal
mcts_traversal = MCTSGraphTraversal(G, weight_ranges, start_node, end_node, k, simulations)

# Minimize the total cost over k traversals
total_cost, paths = mcts_traversal.traverse_k_times()
