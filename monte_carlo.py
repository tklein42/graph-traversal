import random
import math
import networkx as nx

class MCTSNode:
    def __init__(self, node, parent=None):
        self.node = node  # Node ID
        self.parent = parent  # Parent node
        self.children = {}  # Child nodes
        self.visits = 0  # Number of visits
        self.total_cost = 0  # Total cost
    
    def add_child(self, child_node): 
        if child_node.node not in self.children: 
            self.children[child_node.node] = child_node # Add the child node
    
    def update(self, cost):
        self.visits += 1 # Increment visit count
        self.total_cost += cost # Update total cost (Add cost of path including this node)

class MCTSGraphTraversal:
    def __init__(self, graph, weight_ranges, start_node, end_node, k, simulations):
        self.graph = graph # Undirected graph
        self.weight_ranges = weight_ranges # Edge weight ranges
        self.start_node = start_node # Start node
        self.end_node = end_node # End node
        self.k = k # Number of traversals
        self.simulations = simulations # Number of simulations
        self.explored_edges = {} # Keep track of discovered edge weights
    
    def get_edge_weight(self, node1, node2):
        # Check both directions for undirected graphs
        if (node1, node2) in self.explored_edges: 
            return self.explored_edges[(node1, node2)] 
        elif (node2, node1) in self.explored_edges:
            return self.explored_edges[(node2, node1)]
        
        # Sample the weight and store it for both directions
        # Get the min and max weight
        if (node1, node2) in self.weight_ranges:
            min_weight, max_weight = self.weight_ranges[(node1, node2)]
        elif (node2, node1) in self.weight_ranges:
            min_weight, max_weight = self.weight_ranges[(node2, node1)]
        else:
            raise KeyError(f"Edge ({node1}, {node2}) or ({node2}, {node1}) not found in weight_ranges")
        # Sample the weight
        weight = random.uniform(min_weight, max_weight)
        # Store the weight
        self.explored_edges[(node1, node2)] = weight
        self.explored_edges[(node2, node1)] = weight
        return weight
    
    def ucb1(self, node, child, exploration_constant=1.0):
        if child.visits == 0:
            # Encourage exploration for unvisited nodes with a very large value
            return float('inf')
        average_cost = child.total_cost / child.visits # Average cost from each prior visit
        exploration_term = exploration_constant * math.sqrt(math.log(node.visits) / child.visits) # Exploration term
        return -average_cost + exploration_term  # Minimize cost, use -average_cost to maximize reward
    
    def select(self, node):
        current_node = node # Starting node
        while current_node.children: # While there are children
            # Select the child with the highest UCB1 value
            current_node = max(current_node.children.values(), key=lambda child: self.ucb1(current_node, child, 2.0))
        return current_node
    
    def expand(self, node):
        for neighbor in self.graph[node.node]: # For each neighbor
            if neighbor not in node.children: # If it's not already a child
                new_node = MCTSNode(neighbor, node) # Create a new node
                node.add_child(new_node) # Add the new node
    
    def simulate(self, node):
        current_node = node.node # Starting node
        path = [] # List of nodes in the path
        total_cost = 0 # Total cost of the path
        visited = set() # Keep track of visited nodes (Prevent loops in the graph)
        visited.add(current_node) # Add the starting node
        
        while current_node != self.end_node:
            neighbors = list(self.graph[current_node]) # Get the neighbors of the current node
            unvisited_neighbors = [n for n in neighbors if n not in visited] # Filter out visited neighbors
            
            if not unvisited_neighbors: # End simulation there are no unvisited neighbors
                break
            
            next_node = random.choice(unvisited_neighbors) # Select a random unvisited neighbor
            path.append((current_node, next_node)) # Add the edge to the path
            total_cost += self.get_edge_weight(current_node, next_node) # Add the edge weight to the total cost
            visited.add(next_node) # Add the next node to the visited set
            current_node = next_node # Move to the next node
            
            if current_node == self.end_node: # End simulation if the end node is reached
                break
        
        return total_cost, path
    
    def backpropagate(self, node, cost):
        current_node = node # Starting node
        while current_node is not None: # While there is a parent
            current_node.update(cost) # Update the node
            current_node = current_node.parent # Move to the parent
    
    def run_simulation(self):
        root = MCTSNode(self.start_node)  # Ensure each simulation starts from the start_node
        
        for _ in range(self.simulations):
            # Selection
            leaf = self.select(root) # Select a leaf node
            
            # Expansion
            if leaf.node != self.end_node: # If the leaf node is not the end node
                self.expand(leaf) # Expand the leaf node
            
            # Simulation
            simulation_cost, _ = self.simulate(leaf) # Simulate the leaf node
            
            # Backpropagation
            self.backpropagate(leaf, simulation_cost) # Backpropagate the simulation cost
        
        return root
    
    def traverse_k_times(self):
        total_traversal_cost = 0 # Total cost of traversals
        all_paths = [] # List of all paths
        
        for i in range(self.k):
            print(f"Traversal {i+1}/{self.k}")
            root = self.run_simulation() # Run the simulation
            
            # Select the best path (lowest cost) after simulation
            best_path = [] # List of nodes in the best path
            current_node = root # Starting node
            total_cost = 0 # Total cost of the path
            visited = set() # Keep track of visited nodes (Prevent loops in the graph)
            
            while current_node.node != self.end_node: # While the end node is not reached
                visited.add(current_node.node) # Add the current node to the visited set
                
                # If current node has no children, expand it first
                if not current_node.children:
                    self.expand(current_node) # Expand the current node
        
                # Now select the best child using UCB1, favoring exploitation for the traversal
                if current_node.children:
                    # Filter out already visited nodes
                    unvisited_children = [child for child in current_node.children.values() if child.node not in visited]
                    
                    if unvisited_children:
                        # Choose the best child that hasn't been visited yet
                        best_child = max(unvisited_children, key=lambda child: self.ucb1(current_node, child, 0.10))
                    else:
                        # If all children have been visited, stop traversal to avoid cycles
                        break
                    
                    next_node = best_child.node # Get the next node
                    best_path.append((current_node.node, next_node)) # Add the edge to the path
                    total_cost += self.get_edge_weight(current_node.node, next_node) # Add the edge weight to the total cost
                    current_node = best_child # Move to the next node
                else:
                    # Break if somehow can't expand or select any children (If graph somewhow is disconnected)
                    break
                
            total_traversal_cost += total_cost # Add the total cost of the traversal
            all_paths.append(best_path) # Add the best path to the list
            print(f"Traversal Path: {best_path}, Traversal Cost: {total_cost:.2f}")
        
        print(f"Total cost over {self.k} traversals: {total_traversal_cost:.2f}")
        return total_traversal_cost, all_paths