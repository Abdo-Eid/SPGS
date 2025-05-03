# pip install networkx matplotlib numpy gymnasium numba
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import random
import time
import os
from numba import jit, float64, bool_, int64, void
from networkx.algorithms.shortest_paths.generic import shortest_path, has_path
from networkx.algorithms.components import connected_components

# --- Numba Optimized Functions (Keep as is) ---
@jit(float64(float64[:], bool_[:], int64), nopython=True, cache=True)
def compute_served_load(demands, blackout_flags, n_loads):
    total = 0.0
    for i in range(n_loads):
        if not blackout_flags[i]:
            total += demands[i]
    return total

@jit(int64[:](float64[:], float64[:], int64[:], int64), nopython=True, cache=True)
def check_overloads(stresses, capacities, edge_indices, n_edges):
    overloaded = np.zeros(n_edges, dtype=np.int64)
    count = 0
    for i in range(n_edges):
        if capacities[i] != np.inf and stresses[i] > capacities[i]:
            overloaded[count] = edge_indices[i]
            count += 1
    return overloaded[:count]

# --- Save/Load Q-table (Keep as is) ---
Q_TABLE_FILENAME = 'power_grid_q_table_np.npz'
def save_q_table(q_table, state_to_index, next_state_idx, filename):
    try:
        state_keys = np.array([str(k) for k in state_to_index.keys()])
        state_values = np.array(list(state_to_index.values()), dtype=np.int64)
        np.savez(filename,
                 q_table=q_table[:next_state_idx,:],
                 state_keys=state_keys,
                 state_values=state_values,
                 next_state_index=np.array([next_state_idx])
                )
        print(f"Q-table saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving Q-table: {e}")

def load_q_table(filename, action_space_size, state_length):
    initial_q_table_size = 1024
    empty_q_table = np.zeros((initial_q_table_size, action_space_size), dtype=np.float32)
    empty_state_map = {}
    empty_next_idx = 0
    if not os.path.exists(filename):
        print(f"No saved Q-table found at {filename}.")
        return empty_q_table, empty_state_map, empty_next_idx
    try:
        data = np.load(filename, allow_pickle=True)
        loaded_q_table = data['q_table']
        loaded_state_keys = data['state_keys']
        loaded_state_values = data['state_values']
        next_state_idx = int(data['next_state_index'][0])
        if loaded_q_table.shape[1] != action_space_size:
            print(f"Warning: Action space size mismatch! Re-initializing.")
            return empty_q_table, empty_state_map, empty_next_idx
        state_to_index = {}
        for k_str, v_idx in zip(loaded_state_keys, loaded_state_values):
            try: state_tuple = tuple(map(int, k_str.strip('()').split(','))); state_to_index[state_tuple] = v_idx
            except ValueError: print(f"Warning: Could not parse state key '{k_str}'. Skipping."); continue
        current_capacity = max(initial_q_table_size, loaded_q_table.shape[0])
        if current_capacity > loaded_q_table.shape[0]:
            q_table = np.pad(loaded_q_table, ((0, current_capacity - loaded_q_table.shape[0]), (0, 0)), mode='constant')
        else: q_table = loaded_q_table
        print(f"Q-table loaded successfully from {filename} (Size: {q_table.shape}, States: {len(state_to_index)})")
        return q_table, state_to_index, next_state_idx
    except Exception as e:
        print(f"Error loading Q-table: {e}. Returning empty Q-table.")
        return empty_q_table, empty_state_map, empty_next_idx


# --- Grid Topology (Keep as is) ---
G = nx.DiGraph()
# ... (rest of the graph definition remains the same) ...
node_types = {
    'Generator': ['Generator 1 (G1)', 'Generator 2 (G2)'],
    'Substation': ['Substation A', 'Substation B', 'Substation C', 'Substation D', 'Substation E', 'Substation F', 'Substation G'],
    'Load Zone': ['Load Zone 1 (Critical)', 'Load Zone 2', 'Load Zone 3'],
}
for node_type, nodes in node_types.items():
    for node in nodes:
        G.add_node(node, type=node_type)
edges = [
    ('Generator 1 (G1)', 'Substation A', {'label': 'HV Transmission 1', 'capacity_mw': 500, 'in_service': True, 'critical': True, 'component': 'transmission'}),
    ('Generator 2 (G2)', 'Substation F', {'label': 'HV Transmission 2', 'capacity_mw': 400, 'in_service': True, 'critical': True, 'component': 'transmission'}),
    ('Substation A', 'Substation F', {'label': 'Line 10 (Tie)', 'capacity_mw': 300, 'in_service': True, 'critical': True, 'component': 'tie_line'}),
    ('Substation F', 'Substation A', {'label': 'Line 10 (Tie)', 'capacity_mw': 300, 'in_service': True, 'critical': True, 'component': 'tie_line'}),
    ('Substation A', 'Substation B', {'label': 'Line 1', 'capacity_mw': 150, 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation A', 'Substation C', {'label': 'Line 2', 'capacity_mw': 150, 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation F', 'Substation C', {'label': 'Line 8', 'capacity_mw': 100, 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation F', 'Substation G', {'label': 'Line 9', 'capacity_mw': 100, 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation B', 'Substation C', {'label': 'Line 3', 'capacity_mw': 80, 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation B', 'Substation D', {'label': 'Line 4', 'capacity_mw': 70, 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation C', 'Substation E', {'label': 'Line 5', 'capacity_mw': 90, 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation C', 'Substation G', {'label': 'Line 11', 'capacity_mw': 60, 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation G', 'Substation E', {'label': 'Line 12', 'capacity_mw': 70, 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation D', 'Substation E', {'label': 'Line 14', 'capacity_mw': 50, 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation E', 'Substation D', {'label': 'Line 14', 'capacity_mw': 50, 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation D', 'Load Zone 1 (Critical)', {'label': 'Feeder D-LZ1', 'capacity_mw': 100, 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation B', 'Load Zone 1 (Critical)', {'label': 'Feeder B-LZ1', 'capacity_mw': 80, 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation D', 'Load Zone 2', {'label': 'Feeder D-LZ2', 'capacity_mw': 120, 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation G', 'Load Zone 3', {'label': 'Feeder G-LZ3', 'capacity_mw': 90, 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation E', 'Load Zone 3', {'label': 'Feeder E-LZ3', 'capacity_mw': 70, 'in_service': True, 'critical': False, 'component': 'feeder'}),
]
G.add_edges_from(edges)
G.nodes['Load Zone 1 (Critical)']['initial_demand_mw'] = 150
G.nodes['Load Zone 1 (Critical)']['current_demand_mw'] = 150
G.nodes['Load Zone 1 (Critical)']['critical'] = True
G.nodes['Load Zone 2']['initial_demand_mw'] = 80
G.nodes['Load Zone 2']['current_demand_mw'] = 80
G.nodes['Load Zone 2']['critical'] = False
G.nodes['Load Zone 3']['initial_demand_mw'] = 120
G.nodes['Load Zone 3']['current_demand_mw'] = 120
G.nodes['Load Zone 3']['critical'] = False
initial_edge_states = {(u, v): data['in_service'] for u, v, data in G.edges(data=True)}
initial_load_demands = {n: G.nodes[n]['initial_demand_mw'] for n, data in G.nodes(data=True) if data['type'] == 'Load Zone'}
pos = {
    'Generator 1 (G1)': (-3, 7), 'Substation A': (0, 5), 'Substation F': (3, 5),
    'Generator 2 (G2)': (6, 7), 'Substation B': (-2, 3), 'Substation C': (1, 3),
    'Substation G': (4, 3), 'Substation D': (-2, 1), 'Substation E': (3, 1),
    'Load Zone 1 (Critical)': (-3, -1), 'Load Zone 2': (0, -1), 'Load Zone 3': (5, -1),
}
pos = {node: p for node, p in pos.items() if node in G.nodes()}
node_list = list(G.nodes())
node_to_idx = {node: i for i, node in enumerate(node_list)}
idx_to_node = {i: node for i, node in enumerate(node_list)}

# --- Optimized Simulation Logic (Keep as is) ---
def simulate_grid_state(graph, node_mapping=node_to_idx):
    # ... (simulation logic remains the same) ...
    load_nodes = [n for n, data in graph.nodes(data=True) if data['type'] == 'Load Zone']
    n_loads = len(load_nodes)
    blackout_status = {n: True for n in load_nodes}

    if graph.number_of_nodes() == 0:
        return 0.0, blackout_status, set()

    active_generators = [n for n, data in graph.nodes(data=True) if data['type'] == 'Generator']
    operating_edges = [(u, v) for u, v, data in graph.edges(data=True)
                       if data.get('in_service', True) and data.get('component') in ('transmission', 'tie_line', 'line')]

    # Use a copy for component analysis to avoid modifying original graph structure implicitly
    operating_graph = graph.edge_subgraph(operating_edges).copy().to_undirected()
    if operating_graph.number_of_nodes() == 0:
         return 0.0, blackout_status, set()

    gen_components = set()
    components = list(connected_components(operating_graph))
    node_to_component = {}
    for idx, comp in enumerate(components):
        has_generator = False
        for node in comp:
             node_to_component[node] = idx
             if node in active_generators:
                 has_generator = True
        if has_generator:
            gen_components.add(idx)

    for load_node in load_nodes:
        is_connected = False
        feeder_sources = [u for u, v, data in graph.edges(data=True)
                          if v == load_node and data.get('in_service', True) and data.get('component') == 'feeder']
        for feeder_source_node in feeder_sources:
            # Check if feeder source is connected to a generator component
            if feeder_source_node in node_to_component and node_to_component[feeder_source_node] in gen_components:
                is_connected = True
                break # Found a path via this feeder
        if is_connected:
            blackout_status[load_node] = False

    demands = np.array([graph.nodes[n].get('current_demand_mw', 0) for n in load_nodes], dtype=np.float64)
    blackout_flags = np.array([blackout_status[n] for n in load_nodes], dtype=np.bool_)
    served_load_mw = compute_served_load(demands, blackout_flags, n_loads)

    # --- Overload Check ---
    line_stress_load = defaultdict(float)
    # We need the operating graph again for pathfinding (use the undirected version)
    # Ensure operating_graph has nodes before proceeding
    if operating_graph.number_of_nodes() > 0:
        for load_node, is_blacked_out in blackout_status.items():
            if is_blacked_out: continue
            load_demand = graph.nodes[load_node].get('current_demand_mw', 0)
            if load_demand <= 0: continue

            feeder_sources = [u for u, v, data in graph.edges(data=True)
                              if v == load_node and data.get('in_service', True) and data.get('component') == 'feeder']

            path_found_for_load = False
            for feeder_source_node in feeder_sources:
                # Check again if this source is connected to a generator component
                if feeder_source_node in node_to_component and node_to_component[feeder_source_node] in gen_components:
                     comp_idx = node_to_component[feeder_source_node]
                     source_gen = None
                     # Find any generator within the same component
                     for gen in active_generators:
                         if gen in node_to_component and node_to_component[gen] == comp_idx:
                              source_gen = gen
                              break # Found a generator in this component

                     # Check if both nodes exist in the operating_graph before finding path
                     if source_gen and operating_graph.has_node(feeder_source_node) and operating_graph.has_node(source_gen):
                         try:
                             # Find shortest path on the *operating_graph*
                             path = shortest_path(operating_graph, source=feeder_source_node, target=source_gen)
                             # Distribute load along the path edges (check original graph for directionality/existence if needed)
                             for i in range(len(path) - 1):
                                 u_path, v_path = path[i], path[i + 1]
                                 # Check both directions in the original graph for the edge
                                 if graph.has_edge(u_path, v_path) and graph[u_path][v_path].get('component') in ['transmission', 'tie_line', 'line']:
                                     line_stress_load[(u_path, v_path)] += load_demand
                                 # Also add stress if the reverse edge exists and is part of the path conceptually
                                 if graph.has_edge(v_path, u_path) and graph[v_path][u_path].get('component') in ['transmission', 'tie_line', 'line']:
                                      # Be careful not to double-count if the path logic implies direction
                                      # For simple stress, adding to both edges representing the line segment might be okay
                                      # Or, just add to the primary direction found in the path
                                      pass # Avoid double counting for now, assume path direction matters most


                             path_found_for_load = True
                             break # Found a path for this load via this feeder
                         except nx.NetworkXNoPath:
                             # This shouldn't happen if they are in the same component, but handle just in case
                             pass
                         except nx.NodeNotFound:
                             # Handle cases where nodes might not be in operating_graph (shouldn't happen with checks)
                             pass

                if path_found_for_load: break # Move to next load node

    # Check for overloads based on calculated stress
    sim_edge_list = list(line_stress_load.keys())
    n_sim_edges = len(sim_edge_list)
    overloaded_lines = set()
    if n_sim_edges > 0:
        # Ensure edges exist in the main graph before getting data
        valid_edges = [edge for edge in sim_edge_list if graph.has_edge(*edge)]
        if valid_edges:
             n_valid_edges = len(valid_edges)
             stresses = np.array([line_stress_load[edge] for edge in valid_edges], dtype=np.float64)
             capacities = np.array([graph.get_edge_data(u, v).get('capacity_mw', np.inf) for u, v in valid_edges], dtype=np.float64)
             edge_indices_for_numba = np.arange(n_valid_edges, dtype=np.int64) # Indices map to valid_edges list
             overloaded_indices = check_overloads(stresses, capacities, edge_indices_for_numba, n_valid_edges)
             overloaded_lines = {valid_edges[i] for i in overloaded_indices} # Map back to original edge tuples

    return served_load_mw, blackout_status, overloaded_lines


# --- RL Environment ---
class SimplePowerGridEnv(gym.Env):
    initial_q_table_size = 1024

    def __init__(self, nx_graph, initial_edge_states, initial_load_demands, fault_rate=0.05, auto_heal_rate=0.02):
        super().__init__()
        self.nx_graph = nx_graph.copy()
        self._initial_edge_states = initial_edge_states
        self._initial_load_demands = initial_load_demands
        self.fault_rate = fault_rate
        self.auto_heal_rate = auto_heal_rate
        self.current_graph = None
        self.controllable_lines_keys = [(u, v) for u, v, data in self.nx_graph.edges(data=True) if data.get('component') in ['transmission', 'tie_line', 'line']]
        self.controllable_lines_set = set(self.controllable_lines_keys)
        self.sheddable_load_nodes = [n for n, data in self.nx_graph.nodes(data=True) if data['type'] == 'Load Zone' and not data.get('critical', False)]
        self.shed_percentage = 0.2
        self.action_map = {0: ('noop', None)}
        action_idx = 1
        self._line_action_start_idx = action_idx
        for u, v in self.controllable_lines_keys:
            self.action_map[action_idx] = ('open_line', (u, v)); action_idx += 1
            self.action_map[action_idx] = ('close_line', (u, v)); action_idx += 1
        self._shed_action_start_idx = action_idx
        for load_node in self.sheddable_load_nodes:
            self.action_map[action_idx] = ('shed_load', load_node); action_idx += 1
        self.action_space = spaces.Discrete(len(self.action_map))
        self._load_zone_nodes = sorted([n for n, data in self.nx_graph.nodes(data=True) if data['type'] == 'Load Zone'])
        self.state_length = len(self.controllable_lines_keys) + len(self._load_zone_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_length,), dtype=np.int32)
        self.alpha = 0.1
        self.gamma = 0.5
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        self.q_table = None
        self.state_to_index = None
        self.next_state_index = None
        self.current_q_table_capacity = 0
        self.steps_since_last_fault = 0
        self._last_blackout_status = None # Cache for rendering
        self.fig = None # Figure handle
        self.ax = None # Axes handle
        self.pos = pos # Store positions

    # ... (initialize_q_table, _state_to_index_lookup, _get_state_representation) ...
    # ... (_calculate_reward, _update_q_table, choose_action, step, reset) ...
    # Keep these methods as they are

    def initialize_q_table(self, loaded_q_table=None, loaded_state_map=None, loaded_next_idx=None):
         if loaded_q_table is not None and loaded_state_map is not None:
             self.q_table = loaded_q_table
             self.state_to_index = loaded_state_map
             self.next_state_index = loaded_next_idx
             self.current_q_table_capacity = self.q_table.shape[0]
             print(f"Initialized with loaded Q-table (capacity: {self.current_q_table_capacity}, states: {self.next_state_index}).")
         else:
             self.q_table = np.zeros((self.initial_q_table_size, self.action_space.n), dtype=np.float32)
             self.state_to_index = {}
             self.next_state_index = 0
             self.current_q_table_capacity = self.initial_q_table_size
             print(f"Initialized new Q-table (capacity: {self.current_q_table_capacity}).")

    def _state_to_index_lookup(self, state_tuple):
        if state_tuple in self.state_to_index:
            return self.state_to_index[state_tuple]
        else:
            if self.next_state_index >= self.current_q_table_capacity:
                new_capacity = self.current_q_table_capacity * 2
                print(f"Resizing Q-table from {self.current_q_table_capacity} to {new_capacity}")
                # Ensure correct padding: pad rows, not columns
                self.q_table = np.pad(self.q_table, ((0, self.current_q_table_capacity), (0, 0)), mode='constant', constant_values=0)
                self.current_q_table_capacity = new_capacity
            new_index = self.next_state_index
            self.state_to_index[state_tuple] = new_index
            self.next_state_index += 1
            return new_index

    def _get_state_representation(self):
        state_list = []
        # Ensure consistent order matching action map creation if necessary, but here it's just line status
        for u, v in self.controllable_lines_keys:
            # Check if edge exists before accessing attributes
            if self.current_graph.has_edge(u,v):
                 state_list.append(int(self.current_graph[u][v].get('in_service', False)))
            else:
                 # Should not happen if graph structure is static, but good practice
                 state_list.append(0)

        # Get current blackout status if not cached
        if self._last_blackout_status is None:
             _, blackout_status, _ = simulate_grid_state(self.current_graph)
             self._last_blackout_status = blackout_status # Cache it

        for load_node in self._load_zone_nodes: # Use the sorted list
            state_list.append(int(not self._last_blackout_status.get(load_node, True))) # 1 if served, 0 if blacked out

        return tuple(state_list)

    def _calculate_reward(self, served_load_mw, blackout_status, overloaded_lines, action_cost):
        reward = 0
        total_initial_load = sum(self._initial_load_demands.values())

        # Reward for served load (normalized)
        if total_initial_load > 0:
            reward += 100 * (served_load_mw / total_initial_load)
        else:
            reward += 50 # Small reward if there's no load to serve but grid is up

        # Penalties for blackouts
        critical_blackout_penalty = 0
        non_critical_blackout_penalty = 0
        for load_node, is_blacked_out in blackout_status.items():
            if is_blacked_out:
                # Check if node exists and get its properties safely
                node_data = self.current_graph.nodes.get(load_node, {})
                if node_data.get('critical', False):
                    critical_blackout_penalty += 1 # Count critical blackouts
                else:
                    non_critical_blackout_penalty += 1 # Count non-critical blackouts

        # Apply penalties (e.g., higher penalty for critical loads)
        reward -= critical_blackout_penalty * 100 # Heavier penalty for critical zones
        reward -= non_critical_blackout_penalty * 20 # Lighter penalty for non-critical

        # Penalty for overloaded lines
        reward -= 50 * len(overloaded_lines)

        # Cost for taking actions (especially load shedding)
        reward -= action_cost

        return reward

    def _update_q_table(self, state_tuple, action_idx, reward, next_state_tuple, done):
        state_idx = self._state_to_index_lookup(state_tuple)
        next_state_idx = self._state_to_index_lookup(next_state_tuple)

        current_q = self.q_table[state_idx, action_idx]
        # Use max Q value of the *next* state for the update
        max_next_q = np.max(self.q_table[next_state_idx]) if not done else 0.0

        # Bellman equation update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_idx, action_idx] = new_q

    def choose_action(self, state_representation):
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return self.action_space.sample()
        else:
            # Exploit: choose the best action from Q-table
            state_idx = self._state_to_index_lookup(state_representation)
            state_q_values = self.q_table[state_idx]
            # Choose the action index with the highest Q-value
            # Break ties randomly if multiple actions have the same max Q-value
            best_action_indices = np.where(state_q_values == np.max(state_q_values))[0]
            return np.random.choice(best_action_indices)

    def step(self, action_index):
        # --- Get state BEFORE changes for Q-update ---
        current_state_tuple = self._get_state_representation() # Uses cached blackout status if available

        # --- Auto-Healing ---
        healed_lines_info = []
        if self.auto_heal_rate > 0: # Only loop if healing is possible
             # Iterate over a copy of edges keys, as graph might change
             edges_to_check = list(self.current_graph.edges())
             for u, v in edges_to_check:
                 # Check if it's a controllable line and currently out of service
                 if (u, v) in self.controllable_lines_set and self.current_graph.has_edge(u,v) and not self.current_graph[u][v].get('in_service', True):
                     if random.random() < self.auto_heal_rate:
                         # Heal the line
                         self.current_graph[u][v]['in_service'] = True
                         healed_lines_info.append((u, v))
                         # If the reverse edge is also controllable, heal it too (assuming pairs)
                         if (v, u) in self.controllable_lines_set and self.current_graph.has_edge(v, u):
                              if not self.current_graph[v][u].get('in_service', True):
                                  self.current_graph[v][u]['in_service'] = True
                                  # Don't double add to healed_lines_info, handled by reporting logic

        # --- Apply Agent's Action ---
        action_type, action_arg = self.action_map[action_index]
        action_cost = 0.1 # Default small cost for line switching
        if action_type == 'noop':
            action_cost = 0
        elif action_type == 'open_line':
            u, v = action_arg
            if self.current_graph.has_edge(u, v):
                self.current_graph[u][v]['in_service'] = False
                # If the reverse edge is controllable, open it too
                if (v, u) in self.controllable_lines_set and self.current_graph.has_edge(v, u):
                    self.current_graph[v][u]['in_service'] = False
        elif action_type == 'close_line':
            u, v = action_arg
            # Check if the line exists before trying to close it
            if self.current_graph.has_edge(u, v):
                self.current_graph[u][v]['in_service'] = True
                 # If the reverse edge is controllable, close it too
                if (v, u) in self.controllable_lines_set and self.current_graph.has_edge(v, u):
                    self.current_graph[v][u]['in_service'] = True
        elif action_type == 'shed_load':
            load_node = action_arg
            # Ensure the node exists and is a load zone before shedding
            if load_node in self.current_graph.nodes and self.current_graph.nodes[load_node]['type'] == 'Load Zone':
                current_demand = self.current_graph.nodes[load_node]['current_demand_mw']
                shed_amount = current_demand * self.shed_percentage
                self.current_graph.nodes[load_node]['current_demand_mw'] = max(0, current_demand - shed_amount)
                action_cost = shed_amount * 0.5 # Cost proportional to amount shed

        # --- Fault Simulation ---
        faulted_line_key = None
        # Get controllable lines that are currently IN SERVICE
        in_service_controllable_lines = [(u, v) for u, v in self.controllable_lines_keys
                                          if self.current_graph.has_edge(u, v) and self.current_graph[u][v].get('in_service', False)]

        if in_service_controllable_lines and random.random() < self.fault_rate:
            # Choose a random in-service controllable line to fault
            faulted_line_key = random.choice(in_service_controllable_lines)
            u_fault, v_fault = faulted_line_key
            # Take the faulted line out of service
            if self.current_graph.has_edge(u_fault, v_fault):
                 self.current_graph[u_fault][v_fault]['in_service'] = False
            # Take the reverse direction out too if it's controllable
            if (v_fault, u_fault) in self.controllable_lines_set and self.current_graph.has_edge(v_fault, u_fault):
                self.current_graph[v_fault][u_fault]['in_service'] = False

        self.steps_since_last_fault += 1

        # --- Simulate Grid State & Get Next State ---
        self._last_blackout_status = None # Crucial: Invalidate cache before simulation
        served_load_mw, blackout_status, overloaded_lines = simulate_grid_state(self.current_graph)
        self._last_blackout_status = blackout_status # Update cache AFTER simulation
        next_state_tuple = self._get_state_representation() # Get representation AFTER simulation

        # --- Calculate Reward & Done ---
        reward = self._calculate_reward(served_load_mw, blackout_status, overloaded_lines, action_cost)

        # Define 'done' condition (e.g., all critical loads blacked out)
        critical_nodes = [n for n, data in self.current_graph.nodes(data=True)
                           if data.get('critical', False) and data['type'] == 'Load Zone']
        all_critical_blacked_out = False
        if critical_nodes: # Only check if critical nodes exist
             all_critical_blacked_out = all(blackout_status.get(n, True) for n in critical_nodes)
        done = all_critical_blacked_out

        # --- Q-Table Update ---
        self._update_q_table(current_state_tuple, action_index, reward, next_state_tuple, done)

        # --- Info Dictionary ---
        info = {
            'served_load_mw': served_load_mw,
            'blackout_status': blackout_status,
            'overloaded_lines': list(overloaded_lines), # Convert set to list for consistency
            'faulted_line': faulted_line_key,
            'healed_lines': healed_lines_info,
            'action_taken': self.action_map[action_index] # Include action details
        }

        # Standard Gym API return: observation, reward, terminated, truncated, info
        return next_state_tuple, reward, done, False, info # Assuming 'truncated' is always False here

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility if seed is used

        # Create a fresh copy of the base graph for the new episode
        self.current_graph = self.nx_graph.copy()

        # Reset edge states to initial configuration
        for (u, v), initial_status in self._initial_edge_states.items():
            if self.current_graph.has_edge(u, v):
                 self.current_graph[u][v]['in_service'] = initial_status
            # Also reset reverse edge if it exists and mirrors the state (common in models)
            if self.current_graph.has_edge(v, u) and (v,u) in self._initial_edge_states:
                 self.current_graph[v][u]['in_service'] = self._initial_edge_states[(v,u)]


        # Reset load demands to initial values
        for node, initial_demand in self._initial_load_demands.items():
            if node in self.current_graph.nodes():
                 self.current_graph.nodes[node]['current_demand_mw'] = initial_demand

        # Reset episode-specific variables
        self.steps_since_last_fault = 0
        self._last_blackout_status = None # Clear cache for the new episode

        # Get the initial state representation
        initial_state_tuple = self._get_state_representation()

        # Decay epsilon at the start of each episode
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Standard Gym API return: observation, info
        return initial_state_tuple, {} # Return empty info dict on reset

    # --- Live Plotting Methods ---
    def setup_render(self):
        """Initializes the Matplotlib figure and axes for rendering."""
        # Turn on interactive mode BEFORE creating the plot
        if not plt.isinteractive():
             plt.ion()

        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(14, 9))
            self.ax = self.fig.add_subplot(111)
            self.fig.canvas.manager.set_window_title("Power Grid Simulation")
            # Show the figure window immediately but don't block
            plt.show(block=False)
            plt.pause(0.1) # Allow time for window to appear
        else:
             # If figure exists, clear it completely for the new render setup
             self.fig.clf()
             self.ax = self.fig.add_subplot(111)


    def render(self, mode='human', step_info=""):
        """Renders the current grid state using Matplotlib."""
        if mode != 'human' or self.current_graph is None:
            return

        # Ensure the plot window is set up
        if self.fig is None or self.ax is None or not plt.fignum_exists(self.fig.number):
             print("Setting up render window...")
             self.setup_render() # Create fig/ax if needed

        # --- Force Redraw Strategy ---
        # Clear the entire figure before drawing
        self.fig.clf()
        # Add new axes after clearing
        self.ax = self.fig.add_subplot(111)

        # Get current simulation state for coloring/styling
        # Use cached status if available, otherwise simulate
        if self._last_blackout_status is None:
             served_load_mw, blackout_status, overloaded_lines = simulate_grid_state(self.current_graph)
             self._last_blackout_status = blackout_status # Cache it now
        else:
             blackout_status = self._last_blackout_status
             # Still need to recalculate overloads as they depend on current topology/load
             _, _, overloaded_lines = simulate_grid_state(self.current_graph) # Only need overloads here


        # --- Node Drawing ---
        node_colors = {}
        node_shapes = {}
        node_sizes = {}
        node_lists = defaultdict(list)

        for node, data in self.current_graph.nodes(data=True):
            node_type = data['type']
            node_lists[node_type].append(node)

            if node_type == 'Generator':
                node_colors[node], node_shapes[node], node_sizes[node] = 'lightgreen', 's', 3000
            elif node_type == 'Substation':
                node_colors[node], node_shapes[node], node_sizes[node] = 'lightblue', 'o', 2500
            elif node_type == 'Load Zone':
                node_shapes[node], node_sizes[node] = '^', 2000
                is_blacked_out = blackout_status.get(node, True) # Default to True if not found
                is_critical = data.get('critical', False)
                if is_blacked_out:
                    node_colors[node] = 'red'
                elif is_critical:
                    node_colors[node] = 'salmon' # Critical but served
                else:
                    node_colors[node] = 'orange' # Non-critical and served
            else: # Default for unknown types
                node_colors[node], node_shapes[node], node_sizes[node] = 'gray', 'd', 1000

        # Draw nodes by type to handle different shapes correctly
        for node_type, nodes in node_lists.items():
             if not nodes: continue # Skip if no nodes of this type
             type_shape = node_shapes[nodes[0]] # Assume all nodes of a type have the same shape
             nx.draw_networkx_nodes(self.current_graph, self.pos, ax=self.ax, nodelist=nodes,
                                     node_size=[node_sizes[n] for n in nodes],
                                     node_color=[node_colors[n] for n in nodes],
                                     node_shape=type_shape)

        # --- Edge Drawing ---
        edge_colors = []
        edge_widths = []
        edge_labels_dict = {}
        edges_to_draw = list(self.current_graph.edges(data=True)) # Get edges with data

        for u, v, data in edges_to_draw:
            label = data.get('label', f'{u}-{v}')
            in_service = data.get('in_service', False)
            # Check overload status for this specific directed edge (u, v)
            is_overloaded = (u, v) in overloaded_lines

            if not in_service:
                edge_colors.append('red')
                edge_widths.append(1)
                edge_labels_dict[(u, v)] = f"({label} OFF)"
            elif is_overloaded:
                edge_colors.append('purple')
                edge_widths.append(3)
                edge_labels_dict[(u, v)] = f"({label} OVR)"
            else:
                edge_colors.append('gray')
                edge_widths.append(1.5)
                edge_labels_dict[(u, v)] = label # No extra text needed

        # Draw the edges using the collected styles
        # Ensure the edgelist used matches the order of colors/widths
        nx.draw_networkx_edges(self.current_graph, self.pos, ax=self.ax,
                               edgelist=[(u,v) for u,v,_ in edges_to_draw], # Pass just (u,v) tuples
                               edge_color=edge_colors,
                               width=edge_widths,
                               arrowstyle='->', arrowsize=15) # Slightly smaller arrows


        # --- Label Drawing ---
        # Draw node labels (names)
        nx.draw_networkx_labels(self.current_graph, self.pos, ax=self.ax, font_size=9, font_weight='bold')

        # Draw edge labels (status/overload info)
        nx.draw_networkx_edge_labels(self.current_graph, self.pos, ax=self.ax,
                                     edge_labels=edge_labels_dict,
                                     font_color='darkred', font_size=7, alpha=0.9,
                                     label_pos=0.5) # Center labels on edges


        # --- Final Touches ---
        self.ax.set_title(f"Power Grid Simulation\n{step_info}", fontsize=12)
        self.ax.set_axis_off()
        self.fig.tight_layout() # Adjust layout

        # --- Crucial Update Steps ---
        try:
             self.fig.canvas.draw_idle() # Request redraw efficiently
             self.fig.canvas.flush_events() # Process GUI events
        except Exception as e:
             print(f"Error during canvas draw/flush: {e}") # Catch potential GUI errors

        plt.pause(0.01) # VERY IMPORTANT: Small pause allows plot to update


    def close(self):
        """Closes the Matplotlib plot window."""
        if self.fig is not None:
            print("Closing plot window.")
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        # Turn off interactive mode when closing the environment
        # plt.ioff() # Commented out: may interfere if script continues

# --- Training and Demonstration (Keep as is, but ensure setup_render is called correctly) ---
TRAIN_FAULT_RATE = 0.15
AUTO_HEAL_RATE = 0.05
DEMO_FAULT_RATE = 0.25
DEMO_AUTO_HEAL_RATE = 0.10
max_steps_per_episode = 60

# Set interactive mode ON globally before any plotting starts for the demo
plt.ion()

# Initialize env to get sizes
env = SimplePowerGridEnv(G, initial_edge_states, initial_load_demands,
                         fault_rate=TRAIN_FAULT_RATE, auto_heal_rate=AUTO_HEAL_RATE)
action_space_size = env.action_space.n
state_length = env.observation_space.shape[0] # Use observation_space for state length

# Load or initialize Q-table
load_model = input(f"Load trained model from {Q_TABLE_FILENAME}? (y/n): ").lower() == 'y'
loaded_q_table, loaded_state_map, loaded_next_idx = load_q_table(Q_TABLE_FILENAME, action_space_size, state_length)
env.initialize_q_table(loaded_q_table, loaded_state_map, loaded_next_idx)

# Set epsilon
if load_model and env.next_state_index > 0:
    env.epsilon = 0.1 # Low epsilon for demo if loaded
    print(f"Using loaded Q-table. Epsilon set to {env.epsilon} for potential training.")
    perform_training = input("Perform additional training? (y/n): ").lower() == 'y'
else:
    env.epsilon = 1.0 # High epsilon for exploration if starting fresh
    print("Starting training from scratch.")
    perform_training = True

# --- Training loop ---
if perform_training:
    # Turn interactive mode OFF for faster training plots (if desired)
    plt.ioff()

    num_episodes = 1500
    episode_rewards = np.zeros(num_episodes)
    served_load_history = np.zeros(num_episodes)
    overload_count_history = np.zeros(num_episodes)
    blackout_count_history = np.zeros(num_episodes) # Track critical blackouts

    print("\n--- Training ---")
    start_time = time.time()
    for episode in range(num_episodes):
        state_tuple, info = env.reset()
        total_reward = 0
        step_count = 0
        metrics = {'served_load': [], 'overloads': [], 'blackouts': []}

        for step in range(max_steps_per_episode):
            action = env.choose_action(state_tuple)
            next_state_tuple, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state_tuple = next_state_tuple
            step_count += 1

            # Store metrics for averaging
            metrics['served_load'].append(info['served_load_mw'])
            metrics['overloads'].append(len(info['overloaded_lines']))
            # Calculate critical blackouts for this step
            crit_blackouts = sum(1 for node, blacked_out in info['blackout_status'].items()
                                 if blacked_out and env.current_graph.nodes[node].get('critical', False))
            metrics['blackouts'].append(crit_blackouts)

            if done or truncated: # Check both termination conditions
                break

        # Store episode averages/totals
        episode_rewards[episode] = total_reward
        served_load_history[episode] = np.mean(metrics['served_load']) if metrics['served_load'] else 0
        overload_count_history[episode] = np.mean(metrics['overloads']) if metrics['overloads'] else 0
        blackout_count_history[episode] = np.mean(metrics['blackouts']) if metrics['blackouts'] else 0

        # Print progress periodically
        if (episode + 1) % 100 == 0:
            end_time = time.time()
            avg_reward = np.mean(episode_rewards[max(0, episode-99):episode+1])
            avg_load = np.mean(served_load_history[max(0, episode-99):episode+1])
            avg_overload = np.mean(overload_count_history[max(0, episode-99):episode+1])
            avg_blackout = np.mean(blackout_count_history[max(0, episode-99):episode+1])
            time_per_100 = end_time - start_time
            print(f"Ep {episode+1}/{num_episodes}, Avg Rew (100): {avg_reward:.1f}, "
                  f"Avg Load: {avg_load:.1f}, Avg Ovrld: {avg_overload:.2f}, Avg Crit Blkt: {avg_blackout:.2f}, "
                  f"Eps: {env.epsilon:.3f}, States: {env.next_state_index}, Time: {time_per_100:.2f}s")
            start_time = time.time() # Reset timer

    print("\nTraining finished.")

    # --- Plot training results ---
    # Ensure non-interactive mode for static plots
    plt.ioff()
    plt.figure(figsize=(12, 8)); plt.plot(episode_rewards); plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.title("Total Reward per Episode"); plt.grid(True); plt.show()
    plt.figure(figsize=(12, 8)); plt.plot(served_load_history); plt.xlabel("Episode"); plt.ylabel("Average Served Load (MW)"); plt.title("Average Served Load per Episode"); plt.grid(True); plt.show()
    plt.figure(figsize=(12, 8)); plt.plot(overload_count_history); plt.xlabel("Episode"); plt.ylabel("Average Overloaded Lines"); plt.title("Average Overloaded Lines per Episode"); plt.grid(True); plt.show()
    plt.figure(figsize=(12, 8)); plt.plot(blackout_count_history); plt.xlabel("Episode"); plt.ylabel("Avg Critical Blackouts"); plt.title("Average Critical Blacked Out Load Zones per Episode"); plt.grid(True); plt.show()

    save_model_prompt = input(f"Save trained model to {Q_TABLE_FILENAME}? (y/n): ").lower() == 'y'
    if save_model_prompt:
        save_q_table(env.q_table, env.state_to_index, env.next_state_index, Q_TABLE_FILENAME)
else:
    print("Skipping training.")

# --- Demonstration ---
print("\n--- Demonstrating Learned Policy ---")
# Set up demo environment (can reuse 'env' but adjust parameters)
demo_env = env
demo_env.epsilon = 0.0 # No exploration during demo
demo_env.alpha = 0.0 # No learning during demo
demo_env.fault_rate = DEMO_FAULT_RATE
demo_env.auto_heal_rate = DEMO_AUTO_HEAL_RATE
print(f"Demonstration fault rate: {demo_env.fault_rate}, auto-heal rate: {demo_env.auto_heal_rate}")

# Ensure interactive mode is ON for the demo
plt.ion()

# Set up the render window *before* the loop starts
print("Opening demonstration window...")
demo_env.setup_render() # This will create the figure and show it

state_tuple, reset_info = demo_env.reset()

# Render the initial state
initial_served_load, initial_blackout_status, initial_overloaded_lines = simulate_grid_state(demo_env.current_graph)
initial_crit_blackouts = sum(1 for node, blacked_out in initial_blackout_status.items()
                             if blacked_out and demo_env.current_graph.nodes[node].get('critical', False))
initial_info_str = (f"Step: 0, Reward: N/A (Initial State)\n"
                    f"Served Load: {initial_served_load:.2f} MW, Overloads: {len(initial_overloaded_lines)}, "
                    f"Crit Blackouts: {initial_crit_blackouts}")
print("Initial State Render...")
demo_env.render(step_info=initial_info_str)
time.sleep(3) # Pause to view initial state

for step in range(max_steps_per_episode):
    print(f"\n--- Test Step {step+1} ---")

    # Choose action based on policy
    action = demo_env.choose_action(state_tuple) # Epsilon is 0, so it's deterministic
    action_details = demo_env.action_map[action]

    # Print action details
    action_str = f"Action {action}: {action_details[0]}"
    target_info = ""
    if action_details[1] is not None:
        if isinstance(action_details[1], tuple): # Line action
            u, v = action_details[1]
            # Safely get edge data and label
            edge_data = demo_env.current_graph.get_edge_data(u, v, {})
            line_label = edge_data.get('label', f"{u}-{v}")
            target_info = f"Target: {line_label}"
        else: # Load shedding action
            load_node = action_details[1]
            target_info = f"Target: {load_node} (Shed {demo_env.shed_percentage*100:.0f}%)"
    print(f"{action_str} {target_info}")

    # Take step
    next_state_tuple, reward, done, truncated, info = demo_env.step(action)

    # Prepare info string for rendering title
    crit_blackouts = sum(1 for node, blacked_out in info['blackout_status'].items()
                         if blacked_out and demo_env.current_graph.nodes[node].get('critical', False))
    info_str = (f"Step: {step+1}, Action: {action_details[0]}, Reward: {reward:.2f}\n"
                f"Served Load: {info['served_load_mw']:.2f} MW, Overloads: {len(info['overloaded_lines'])}, "
                f"Crit Blackouts: {crit_blackouts}")

    # Add fault/heal info
    faulted_line_key = info.get('faulted_line')
    healed_lines_info = info.get('healed_lines')
    if faulted_line_key:
         u,v = faulted_line_key
         edge_data = demo_env.current_graph.get_edge_data(u, v, {}) # Get data from current state (it's off now)
         fault_label = edge_data.get('label', faulted_line_key)
         info_str += f"\nFAULT: {fault_label} tripped!"
         print(f"FAULT: {fault_label} tripped!")
    if healed_lines_info:
        healed_labels = []
        processed_pairs = set()
        for u,v in healed_lines_info:
            pair = tuple(sorted((u, v))) # Treat (u,v) and (v,u) as the same line for reporting
            if pair not in processed_pairs:
                 # Get edge data (it should be 'in_service': True now)
                 edge_data = demo_env.current_graph.get_edge_data(u, v, {})
                 label = edge_data.get('label', (u,v))
                 healed_labels.append(label)
                 processed_pairs.add(pair)
        if healed_labels:
             heal_str = f"\nHEALED: {', '.join(map(str, healed_labels))}"
             info_str += heal_str
             print(heal_str)

    # Render the current state
    demo_env.render(step_info=info_str)
    time.sleep(2.5) # Pause between steps

    state_tuple = next_state_tuple # Update state for next iteration

    if done or truncated:
        print(f"Episode finished at step {step+1}. Reason: {'All critical loads out' if done else 'Truncated'}")
        final_info_str = info_str + f"\nEpisode {'Done!' if done else 'Truncated!'}"
        demo_env.render(step_info=final_info_str)
        time.sleep(5) # Longer pause at the end
        break

print("Demonstration finished.")
# Explicitly close the environment window
demo_env.close()
print("Plot window closed.")

# Optional: Turn interactive mode off if the script ends here
plt.ioff()