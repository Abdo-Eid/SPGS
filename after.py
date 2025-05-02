import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import random
import time
import matplotlib.animation as animation
import pickle # Import the pickle module
import os # To check if a file exists

# Set matplotlib to interactive mode for live updates
plt.ion()

# Define filename for saving/loading the Q-table
Q_TABLE_FILENAME = 'power_grid_q_table.pkl'

# --- Functions for Saving and Loading the Q-table ---

def save_q_table(q_table, filename):
    """Saves the Q-table to a file using pickle."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(dict(q_table), f) # Convert defaultdict to dict for saving
        print(f"Q-table saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving Q-table: {e}")

def load_q_table(filename):
    """Loads the Q-table from a file using pickle."""
    # Need a way to get action_space_size here if loading before creating env
    # A robust way is to save it with the q_table. For simplicity, let's assume env is created first.
    # However, the dummy_env trick below handles getting action space size early.
    # Let's pass action_space_size to this function.
    # This requires a slight restructuring of the main loading logic.
    print("Loading Q-table might require knowing the action space size beforehand.")
    print("Proceeding assuming the environment structure matches.")


    if not os.path.exists(filename):
        print(f"No saved Q-table found at {filename}.")
        # We need action_space_size here, which is tricky if env isn't ready.
        # The current approach of getting it from a dummy env is okay for now.
        # If this function is called before any env is created, it would fail.
        # A robust fix needs `action_space_size` passed in or saved in the file.
        # For now, let's rely on the dummy_env trick before calling this.
        # Returning an empty defaultdict which will be updated later
        return defaultdict(lambda: np.zeros(1)) # Dummy size 1, will be overwritten

    try:
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
            q_table = defaultdict(lambda: np.zeros(len(next(iter(loaded_dict.values()))))) # Infer action space size from loaded data
            q_table.update(loaded_dict)
        print(f"Q-table loaded successfully from {filename}")
        return q_table
    except Exception as e:
        print(f"Error loading Q-table: {e}. Returning empty Q-table.")
        # Again, tricky without action_space_size.
        # Returning an empty defaultdict that will adapt when used.
        # The first state it encounters will define the shape.
        # This might be brittle if the first state during training/demo doesn't cover all actions.
        # The safest is to pass action_space_size OR save it with the table.
        # Let's stick to the dummy env trick to get action_space_size reliably before loading.
        return defaultdict(lambda: np.zeros(1))


# --- 1. Create the Grid Topology using NetworkX ---
# (Existing code for creating G, node_types, edges, initial_states, pos)
# ... (Copy all the code for defining G, edges, initial_states, pos here) ...

G = nx.DiGraph()
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

initial_edge_states = { (u, v): data['in_service'] for u, v, data in G.edges(data=True) }
initial_load_demands = { n: G.nodes[n]['initial_demand_mw'] for n, data in G.nodes(data=True) if data['type'] == 'Load Zone' }

pos = {
    'Generator 1 (G1)': (-3, 7), 'Substation A': (0, 5), 'Substation F': (3, 5),
    'Generator 2 (G2)': (6, 7), 'Substation B': (-2, 3), 'Substation C': (1, 3),
    'Substation G': (4, 3), 'Substation D': (-2, 1), 'Substation E': (3, 1),
    'Load Zone 1 (Critical)': (-3, -1), 'Load Zone 2': (0, -1), 'Load Zone 3': (5, -1),
}
pos = {node: p for node, p in pos.items() if node in G.nodes()}


# --- 2. Simplified Simulation Logic (remains the same) ---
# ... (Copy simulate_grid_state function here) ...
def simulate_grid_state(graph):
    """
    Simulates grid state based on connectivity and simplified loading heuristic.
    Returns served_load, blackout_status, overloaded_lines.
    """
    served_load_mw = 0
    blackout_status = {n: True for n, data in graph.nodes(data=True) if data['type'] == 'Load Zone'}
    overloaded_lines = set()

    active_generators = [n for n, data in graph.nodes(data=True) if data['type'] == 'Generator' ]

    operating_edges = [(u, v) for u, v, data in graph.edges(data=True)
                       if data.get('in_service', True) and data.get('component') in ['transmission', 'tie_line', 'line']]
    operating_graph = graph.edge_subgraph(operating_edges).to_undirected()

    for load_node, load_data in graph.nodes(data=True):
        if load_data['type'] == 'Load Zone':
            is_connected = False
            feeder_sources = [u for u, v, data in graph.edges(data=True)
                              if v == load_node and data.get('in_service', True) and data.get('component') == 'feeder']

            if operating_graph.number_of_nodes() > 0 and feeder_sources:
                for gen_node in active_generators:
                    if operating_graph.has_node(gen_node):
                        for feeder_source_node in feeder_sources:
                             if operating_graph.has_node(feeder_source_node):
                                if nx.has_path(operating_graph, source=gen_node, target=feeder_source_node):
                                    is_connected = True
                                    break
                    if is_connected:
                        break

            if is_connected:
                blackout_status[load_node] = False
                served_load_mw += load_data.get('current_demand_mw', 0)

    line_stress_load = defaultdict(float)
    for load_node, is_blacked_out in blackout_status.items():
        if not is_blacked_out:
            load_demand = graph.nodes[load_node].get('current_demand_mw', 0)
            if load_demand > 0:
                feeder_sources = [u for u, v, data in graph.edges(data=True)
                                  if v == load_node and data.get('in_service', True) and data.get('component') == 'feeder']
                if operating_graph.number_of_nodes() > 0 and feeder_sources:
                     for feeder_source_node in feeder_sources:
                          if operating_graph.has_node(feeder_source_node):
                                for gen_node in active_generators:
                                     if operating_graph.has_node(gen_node):
                                          try:
                                             for path in nx.all_simple_paths(operating_graph, source=feeder_source_node, target=gen_node, cutoff=6):
                                                 for i in range(len(path) - 1):
                                                     u, v = path[i], path[i+1]
                                                     if G.has_edge(u, v) and G[u][v].get('component') in ['transmission', 'tie_line', 'line']:
                                                          line_stress_load[(u, v)] += load_demand / len(path)
                                                     if G.has_edge(v, u) and G[v][u].get('component') in ['transmission', 'tie_line', 'line']:
                                                           line_stress_load[(v, u)] += load_demand / len(path)

                                          except nx.NetworkXNoPath:
                                             pass

    for (u, v), stress in line_stress_load.items():
        edge_data = graph.get_edge_data(u, v)
        if edge_data:
             capacity = edge_data.get('capacity_mw', float('inf'))
             if stress > capacity:
                 overloaded_lines.add((u, v))

    return served_load_mw, blackout_status, overloaded_lines


# --- 3. Define the RL Environment using gymnasium ---
# ... (Copy SimplePowerGridEnv class here, including setup_render and render modifications) ...

class SimplePowerGridEnv(gym.Env):
    def __init__(self, nx_graph, initial_edge_states, initial_load_demands, fault_rate=0.05, auto_heal_rate=0.02):
        super(SimplePowerGridEnv, self).__init__()

        self.nx_graph = nx_graph.copy()
        self._initial_edge_states = initial_edge_states
        self._initial_load_demands = initial_load_demands
        self.fault_rate = fault_rate
        self.auto_heal_rate = auto_heal_rate # Probability a downed line auto-heals per step

        self.current_graph = None

        self.controllable_lines_keys = [
            (u, v) for u, v, data in self.nx_graph.edges(data=True)
            if data.get('component') in ['transmission', 'tie_line', 'line']
        ]

        self.sheddable_load_nodes = [
             n for n, data in self.nx_graph.nodes(data=True)
             if data['type'] == 'Load Zone' and not data.get('critical', False)
        ]
        self.shed_percentage = 0.2

        num_controllable_lines = len(self.controllable_lines_keys)
        num_sheddable_loads = len(self.sheddable_load_nodes)

        self.action_map = {0: ('noop', None)}
        action_idx = 1
        self._line_action_start_idx = action_idx
        for i, (u, v) in enumerate(self.controllable_lines_keys):
            self.action_map[action_idx] = ('open_line', (u, v))
            action_idx += 1
            self.action_map[action_idx] = ('close_line', (u, v))
            action_idx += 1
        self._shed_action_start_idx = action_idx
        for j, load_node in enumerate(self.sheddable_load_nodes):
            self.action_map[action_idx] = ('shed_load', load_node)
            action_idx += 1

        self.action_space = spaces.Discrete(len(self.action_map))

        num_state_lines = len(self.controllable_lines_keys)
        num_state_loads = len([n for n, data in self.nx_graph.nodes(data=True) if data['type'] == 'Load Zone'])
        self.state_length = num_state_lines + num_state_loads
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_length,), dtype=int)

        self._load_zone_nodes = sorted([n for n, data in self.nx_graph.nodes(data=True) if data['type'] == 'Load Zone'])

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.q_table = defaultdict(lambda: np.zeros(self.action_space.n))

        self.steps_since_last_fault = 0

        # Attributes for live plotting
        self.fig = None
        self.ax = None
        self.pos = pos # Use the global pos dictionary


    def _get_state_representation(self):
        state_list = []
        for u, v in self.controllable_lines_keys:
            state_list.append(int(self.current_graph[u][v].get('in_service', False)))

        _, blackout_status, _ = simulate_grid_state(self.current_graph)
        for load_node in self._load_zone_nodes:
             state_list.append(int(not blackout_status.get(load_node, True)))

        return tuple(state_list)

    def _calculate_reward(self, served_load_mw, blackout_status, overloaded_lines, action_cost):
        reward = 0
        total_initial_load = sum(self._initial_load_demands.values())
        if total_initial_load > 0:
            reward += 100 * (served_load_mw / total_initial_load)

        for load_node, is_blacked_out in blackout_status.items():
            if is_blacked_out:
                penalty = 100
                if self.current_graph.nodes[load_node].get('critical', False):
                     penalty *= 5
                reward -= penalty

        reward -= 50 * len(overloaded_lines)
        reward -= action_cost

        return reward

    def step(self, action_index):
        if action_index < 0 or action_index >= self.action_space.n:
             action_index = 0

        action_type, action_arg = self.action_map[action_index]
        action_cost = 0.1

        # --- Add Auto-Healing Logic ---
        healed_lines_info = []
        # Iterate over a copy of edges or keys to avoid issues if graph is modified during iteration
        for u, v in list(self.current_graph.edges()):
             # Only attempt to heal controllable lines that are out of service
             if (u, v) in self.controllable_lines_keys and not self.current_graph[u][v].get('in_service', True):
                 if random.random() < self.auto_heal_rate:
                     self.current_graph[u][v]['in_service'] = True
                     healed_lines_info.append((u, v))
                     # Also heal the reverse edge if it exists, is controllable, and is out of service
                     reverse_edge = (v, u)
                     if reverse_edge in self.controllable_lines_keys and self.current_graph.has_edge(v, u) and not self.current_graph[v][u].get('in_service', True):
                          self.current_graph[v][u]['in_service'] = True
                          # Don't add reverse edge to healed_lines_info if we just want to report the 'line' once
                          # healed_lines_info.append((v, u))

        # Remove reverse edges from info if the forward edge was already added
        # This prevents reporting bidirectional lines as healing twice
        # healed_lines_set = set(healed_lines_info)
        # unique_healed_lines_info = []
        # for u,v in healed_lines_info:
        #     if (v,u) not in unique_healed_lines_info: # Simple check
        #         unique_healed_lines_info.append((u,v))
        # healed_lines_info = unique_healed_lines_info # Update the list


        # --- Apply Agent's Action (after healing check) ---
        if action_type == 'noop':
            action_cost = 0
            pass
        elif action_type == 'open_line':
            u, v = action_arg
            if self.current_graph.has_edge(u, v):
                if (u, v) in self.controllable_lines_keys:
                    self.current_graph[u][v]['in_service'] = False
                    reverse_edge = (v, u)
                    if reverse_edge in self.controllable_lines_keys:
                         if self.current_graph.has_edge(v, u):
                             self.current_graph[v][u]['in_service'] = False

        elif action_type == 'close_line':
            u, v = action_arg
            if self.current_graph.has_edge(u, v):
                 if (u, v) in self.controllable_lines_keys:
                     self.current_graph[u][v]['in_service'] = True
                     reverse_edge = (v, u)
                     if reverse_edge in self.controllable_lines_keys:
                          if self.current_graph.has_edge(v, u):
                              self.current_graph[v][u]['in_service'] = True

        elif action_type == 'shed_load':
            load_node = action_arg
            if load_node in self.current_graph.nodes and self.current_graph.nodes[load_node]['type'] == 'Load Zone':
                 current_demand = self.current_graph.nodes[load_node]['current_demand_mw']
                 shed_amount = current_demand * self.shed_percentage
                 new_demand = max(0, current_demand - shed_amount)
                 self.current_graph.nodes[load_node]['current_demand_mw'] = new_demand
                 action_cost = shed_amount * 0.5

        # --- Simulate New Fault (after healing and action) ---
        faulted_line_key = None
        in_service_controllable_lines = [
             (u, v) for u, v in self.controllable_lines_keys
             if self.current_graph.has_edge(u, v) and self.current_graph[u][v].get('in_service', False)
        ]

        if in_service_controllable_lines and random.random() < self.fault_rate:
             # Choose a random IN-SERVICE controllable line to fault
             faulted_line_key = random.choice(in_service_controllable_lines)
             u_fault, v_fault = faulted_line_key
             if self.current_graph.has_edge(u_fault, v_fault):
                 self.current_graph[u_fault][v_fault]['in_service'] = False
                 reverse_edge = (v_fault, u_fault)
                 if reverse_edge in self.controllable_lines_keys:
                      if self.current_graph.has_edge(v_fault, u_fault):
                           self.current_graph[v_fault][u_fault]['in_service'] = False


        self.steps_since_last_fault += 1

        # --- Simulate Grid State after healing, action, and potential new fault ---
        served_load_mw, blackout_status, overloaded_lines = simulate_grid_state(self.current_graph)

        # --- Get Next State Representation ---
        next_state_representation = self._get_state_representation()

        # --- Calculate Reward ---
        reward = self._calculate_reward(served_load_mw, blackout_status, overloaded_lines, action_cost)

        # --- Determine if Done ---
        all_critical_blacked_out = all(blackout_status.get(n, True) for n, data in self.current_graph.nodes(data=True) if data.get('critical', False) and data['type'] == 'Load Zone')
        done = all_critical_blacked_out

        # Info dictionary - include healed lines
        info = {
            'served_load_mw': served_load_mw,
            'blackout_status': blackout_status,
            'overloaded_lines': list(overloaded_lines),
            'faulted_line': faulted_line_key,
            'healed_lines': healed_lines_info, # Add info about lines that healed
        }

        current_state_representation = self._get_state_representation()
        self._update_q_table(current_state_representation, action_index, reward, next_state_representation, done)


        return next_state_representation, reward, done, False, info

    def _update_q_table(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def choose_action(self, state_representation):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            state_q_values = self.q_table[state_representation]
            max_q = np.max(state_q_values)
            best_actions = np.where(state_q_values == max_q)[0]
            return random.choice(best_actions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_graph = self.nx_graph.copy()

        for (u, v), initial_status in self._initial_edge_states.items():
            if self.current_graph.has_edge(u, v):
                 self.current_graph[u][v]['in_service'] = initial_status

        for node, initial_demand in self._initial_load_demands.items():
            if node in self.current_graph.nodes():
                 self.current_graph.nodes[node]['current_demand_mw'] = initial_demand

        self.steps_since_last_fault = 0

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        initial_state = self._get_state_representation()

        info = {}
        return initial_state, info

    # --- Live Plotting Methods ---
    def setup_render(self):
        if not hasattr(self, 'fig') or self.fig is None or not plt.get_fignums():
             self.fig, self.ax = plt.subplots(figsize=(14, 9))
             plt.show(block=False)
        self.ax.cla()


    def render(self, mode='human', highlight_edges=None, step_info=""):
        if mode == 'human':
            self.setup_render()

            served_load_mw, blackout_status, overloaded_lines = simulate_grid_state(self.current_graph)

            node_colors = []
            for node, data in self.current_graph.nodes(data=True):
                if data['type'] == 'Generator': node_colors.append('lightgreen')
                elif data['type'] == 'Substation': node_colors.append('lightblue')
                elif data['type'] == 'Load Zone':
                    if blackout_status.get(node, True): node_colors.append('red')
                    elif data.get('critical', False): node_colors.append('salmon')
                    else: node_colors.append('orange')
                else: node_colors.append('gray')

            edge_colors = []
            edge_widths = []
            edge_labels_dict = {}

            for u, v, data in self.current_graph.edges(data=True):
                edge_label = data.get('label', f'{u}-{v}')
                in_service = data.get('in_service', False)
                is_overloaded = (u, v) in overloaded_lines
                is_highlighted = highlight_edges and ((u,v) in highlight_edges or (v,u) in highlight_edges)

                if not in_service:
                    edge_colors.append('red')
                    edge_widths.append(1)
                    edge_labels_dict[(u, v)] = f"({edge_label} OFF)"
                elif is_overloaded:
                    edge_colors.append('purple')
                    edge_widths.append(3)
                    edge_labels_dict[(u, v)] = f"({edge_label} OVR)"
                elif is_highlighted:
                     edge_colors.append('green')
                     edge_widths.append(2.5)
                     edge_labels_dict[(u,v)] = edge_label
                else:
                    edge_colors.append('gray')
                    edge_widths.append(1.5)
                    edge_labels_dict[(u,v)] = edge_label

            gen_nodes = [n for n,d in self.current_graph.nodes(data=True) if d['type']=='Generator']
            sub_nodes = [n for n,d in self.current_graph.nodes(data=True) if d['type']=='Substation']
            load_nodes = [n for n,d in self.current_graph.nodes(data=True) if d['type']=='Load Zone']

            nx.draw_networkx_nodes(self.current_graph, self.pos, ax=self.ax, nodelist=gen_nodes, node_size=3000, node_color='lightgreen', node_shape='s')
            nx.draw_networkx_nodes(self.current_graph, self.pos, ax=self.ax, nodelist=sub_nodes, node_size=2500, node_color='lightblue', node_shape='o')
            nx.draw_networkx_nodes(self.current_graph, self.pos, ax=self.ax, nodelist=load_nodes, node_size=2000,
                                   node_color=[node_colors[list(self.current_graph.nodes()).index(n)] for n in load_nodes],
                                   node_shape='^')

            nx.draw_networkx_edges(self.current_graph, self.pos, ax=self.ax, edge_color=edge_colors, width=edge_widths, arrowstyle='->', arrowsize=20)
            nx.draw_networkx_labels(self.current_graph, self.pos, ax=self.ax, font_size=9, font_weight='bold')
            nx.draw_networkx_edge_labels(self.current_graph, self.pos, ax=self.ax, edge_labels=edge_labels_dict, font_color='darkred', font_size=7, alpha=0.8)

            self.ax.set_title(f"Complex Power Grid State\n{step_info}")
            self.ax.set_axis_off()
            plt.tight_layout()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        plt.ioff()


# --- 4. Training Loop with Load/Save ---

# Increased fault rate for training
TRAIN_FAULT_RATE = 0.15
AUTO_HEAL_RATE = 0.20 # 2% chance for each downed line to heal per step during training

# Instantiate a dummy environment initially to get action space size for Q-table
dummy_env = SimplePowerGridEnv(G, initial_edge_states, initial_load_demands, fault_rate=TRAIN_FAULT_RATE, auto_heal_rate=AUTO_HEAL_RATE)
action_space_size = dummy_env.action_space.n

# --- Ask the user if they want to load ---
load_model = input(f"Load trained model from {Q_TABLE_FILENAME}? (y/n): ").lower() == 'y'

# --- Load or Initialize Q-table ---
q_table = defaultdict(lambda: np.zeros(action_space_size)) # Initialize with correct size
if load_model:
     # Load into the correctly sized defaultdict
     q_table = load_q_table(Q_TABLE_FILENAME)
     # Ensure loaded Q-table has the correct action space size (handle potential mismatch)
     # This is a basic check; a real system would need versioning or more robust checks
     if q_table and len(next(iter(q_table.values()))) != action_space_size:
          print(f"Warning: Loaded Q-table action space size mismatch! ({len(next(iter(q_table.values())))} vs {action_space_size}). Re-initializing Q-table.")
          q_table = defaultdict(lambda: np.zeros(action_space_size)) # Reset if mismatch

# Create the environment instance
env = SimplePowerGridEnv(G, initial_edge_states, initial_load_demands, fault_rate=TRAIN_FAULT_RATE, auto_heal_rate=AUTO_HEAL_RATE)
env.q_table = q_table # Assign the loaded or newly initialized Q-table

# Set epsilon based on whether loaded or training from scratch
if load_model and len(q_table) > 0: # Check if q_table actually has learned states
    env.epsilon = 0.1 # Lower epsilon for loaded models
    print(f"Using loaded Q-table. Epsilon set to {env.epsilon}")
    perform_training = input("Perform additional training? (y/n): ").lower() == 'y'
else:
    env.epsilon = 1.0 # Start with full exploration if new/empty
    print("Starting training from scratch.")
    perform_training = True # Always train if starting from scratch


max_steps_per_episode = 50
# --- Perform Training if requested ---
if perform_training:
    num_episodes = 1000 # Adjust based on desired training time

    print("\n--- Training ---")
    episode_rewards = []
    served_load_history = []
    overload_count_history = []
    blackout_count_history = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        served_load_this_episode = []
        overload_count_this_episode = []
        blackout_count_this_episode = []

        for step in range(max_steps_per_episode):
            action = env.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            served_load_this_episode.append(info['served_load_mw'])
            overload_count_this_episode.append(len(info['overloaded_lines']))
            blackout_count_this_episode.append(sum(info['blackout_status'].get(n, True) for n, data in env.current_graph.nodes(data=True) if data.get('critical', False) and data['type'] == 'Load Zone'))
            if done: break

        episode_rewards.append(total_reward)
        served_load_history.append(np.mean(served_load_this_episode) if served_load_this_episode else 0)
        overload_count_history.append(np.mean(overload_count_this_episode) if overload_count_this_episode else 0)
        blackout_count_history.append(np.mean(blackout_count_this_episode) if blackout_count_this_episode else 0)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_load = np.mean(served_load_history[-100:])
            avg_overload = np.mean(overload_count_history[-100:])
            avg_blackout = np.mean(blackout_count_history[-100:])
            q_table_size = len(env.q_table)
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 100): {avg_reward:.2f}, Avg Load Served: {avg_load:.2f} MW, Avg Overloads: {avg_overload:.2f}, Avg Critical Blackouts: {avg_blackout:.2f}, Epsilon: {env.epsilon:.2f}, Q-table size: {q_table_size}")

    print("\nTraining finished.")

    # --- Ask the user if they want to save ---
    save_model_prompt = input(f"Save trained model to {Q_TABLE_FILENAME}? (y/n): ").lower() == 'y'
    if save_model_prompt:
        save_q_table(env.q_table, Q_TABLE_FILENAME)

    # Plot training results if training was performed
    plt.ioff()
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode (Q-Learning)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(served_load_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Served Load (MW)")
    plt.title("Average Served Load per Episode")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(overload_count_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Overloaded Lines")
    plt.title("Average Overloaded Lines per Episode (Simplified Heuristic)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(blackout_count_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Critical Blacked Out Load Zones")
    plt.title("Average Critical Blacked Out Load Zones per Episode")
    plt.grid(True)
    plt.show()


else:
    # If not performing training, skip plotting training results
    print("Skipping training plots.")


# --- 6. Demonstrate Learned Policy ---
print("\n--- Demonstrating Learned Policy ---")

# Use the 'env' instance which either has loaded or newly trained Q-table
demo_env = env # Use the same environment instance
# Set epsilon to 0 for demonstration to only use learned policy
demo_env.epsilon = 0.0
demo_env.alpha = 0.0 # No further learning during demo
# Set fault and heal rates for the demo (can be different from training)
DEMO_FAULT_RATE = 0.25 # Higher fault rate for demo
DEMO_AUTO_HEAL_RATE = 0.15 # Higher heal rate for demo
demo_env.fault_rate = DEMO_FAULT_RATE
demo_env.auto_heal_rate = DEMO_AUTO_HEAL_RATE
print(f"Demonstration fault rate: {demo_env.fault_rate}, auto-heal rate: {demo_env.auto_heal_rate}")


# Setup rendering for the demo
demo_env.setup_render()

# Reset the environment for the demo episode
obs, reset_info = demo_env.reset()

# Manually simulate the state to get the initial metrics after reset
initial_served_load, initial_blackout_status, initial_overloaded_lines = simulate_grid_state(demo_env.current_graph)

initial_info_str = (f"Step: 0, Reward: N/A (Initial State)\n"
                    f"Served Load: {initial_served_load:.2f} MW, "
                    f"Overloads: {len(initial_overloaded_lines)}, "
                    f"Blackouts (Crit): {sum(initial_blackout_status.get(n, True) for n, data in demo_env.current_graph.nodes(data=True) if data.get('critical', False) and data['type'] == 'Load Zone')}")

print("Initial State:")
demo_env.render(step_info=initial_info_str)
time.sleep(2)

for step in range(max_steps_per_episode):
    print(f"\n--- Test Step {step+1} ---")
    current_state_rep = demo_env._get_state_representation()
    action = demo_env.choose_action(current_state_rep)
    action_details = demo_env.action_map[action]

    action_str = f"Action {action}: {action_details[0]}"
    if action_details[1] is not None:
         if isinstance(action_details[1], tuple):
              u, v = action_details[1]
              line_label = demo_env.current_graph.get_edge_data(u,v,{}).get('label', f"{u}-{v}")
              action_str += f" Target: {line_label}"
         else:
              load_node = action_details[1]
              action_str += f" Target: {load_node} (Shed {demo_env.shed_percentage*100:.0f}%)"
    print(action_str)

    next_obs, reward, done, truncated, info = demo_env.step(action)

    info_str = (f"Step: {step+1}, Reward: {reward:.2f}\n"
                f"Served Load: {info['served_load_mw']:.2f} MW, "
                f"Overloads: {len(info['overloaded_lines'])}, "
                f"Blackouts (Crit): {sum(info['blackout_status'].get(n, True) for n, data in demo_env.current_graph.nodes(data=True) if data.get('critical', False) and data['type'] == 'Load Zone')})")

    faulted_lines_info = info.get('faulted_line')
    healed_lines_info = info.get('healed_lines')

    if faulted_lines_info:
         u,v = faulted_lines_info
         fault_label = demo_env.current_graph.get_edge_data(u,v,{}).get('label', faulted_lines_info)
         info_str += f"\nFAULT: {fault_label} tripped!"

    if healed_lines_info:
        healed_labels = []
        for u,v in healed_lines_info:
             label = demo_env.current_graph.get_edge_data(u,v,{}).get('label', (u,v))
             if label not in healed_labels: # Avoid adding label twice for bidirectional
                  healed_labels.append(label)
        if healed_labels:
             info_str += f"\nHEALED: {', '.join(map(str, healed_labels))}"


    demo_env.render(step_info=info_str)
    time.sleep(1.5)

    if done:
        print("Episode finished.")
        demo_env.render(step_info=info_str + "\nEpisode Done!")
        time.sleep(4)
        break


# Close the demo plot at the very end
demo_env.close()