import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Import necessary components from other modules
from grid_simulation import simulate_grid_state
# Q-table handling is managed by the main script, but utils might be needed elsewhere
# from grid_utils import save_q_table, load_q_table

class SimplePowerGridEnv(gym.Env):
    """
    A Gymnasium environment for simulating power grid control using Reinforcement Learning.

    The agent learns to manage line switching and load shedding to maximize served load
    while minimizing blackouts and line overloads, considering random faults and auto-healing.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4} # Add metadata if using renderers

    initial_q_table_size = 1024 # Default starting size for Q-table rows

    def __init__(self, nx_graph, initial_edge_states, initial_load_demands, fault_rate=0.05, auto_heal_rate=0.02):
        """
        Initializes the power grid environment.

        Args:
            nx_graph (nx.DiGraph): The base NetworkX graph of the power grid.
            initial_edge_states (dict): {(u, v): bool} Initial 'in_service' status of edges.
            initial_load_demands (dict): {node: float} Initial demand for load zone nodes.
            fault_rate (float): Probability of a line fault occurring in each step.
            auto_heal_rate (float): Probability of an out-of-service line healing in each step.
        """
        super().__init__()

        # --- Grid Configuration ---
        self.nx_graph = nx_graph.copy() # Store a copy of the base graph
        self._initial_edge_states = initial_edge_states
        self._initial_load_demands = initial_load_demands
        self.current_graph = None # Holds the graph state for the current episode

        # --- Environment Dynamics ---
        self.fault_rate = fault_rate
        self.auto_heal_rate = auto_heal_rate
        self.steps_since_last_fault = 0

        # --- Action Space Definition ---
        # Identify controllable lines (transmission, tie, regular lines)
        self.controllable_lines_keys = sorted([(u, v) for u, v, data in self.nx_graph.edges(data=True)
                                              if data.get('component') in ['transmission', 'tie_line', 'line']])
        self.controllable_lines_set = set(self.controllable_lines_keys) # For quick lookups

        # Identify sheddable load nodes (non-critical load zones)
        self.sheddable_load_nodes = sorted([n for n, data in self.nx_graph.nodes(data=True)
                                            if data['type'] == 'Load Zone' and not data.get('critical', False)])
        self.shed_percentage = 0.2 # Percentage of current demand to shed per action

        # Create mapping from action index to action type and argument
        self.action_map = {0: ('noop', None)}
        action_idx = 1
        # Line switching actions (open/close for each controllable line)
        self._line_action_start_idx = action_idx
        for u, v in self.controllable_lines_keys:
            self.action_map[action_idx] = ('open_line', (u, v)); action_idx += 1
            self.action_map[action_idx] = ('close_line', (u, v)); action_idx += 1
        # Load shedding actions (one for each sheddable load zone)
        self._shed_action_start_idx = action_idx
        for load_node in self.sheddable_load_nodes:
            self.action_map[action_idx] = ('shed_load', load_node); action_idx += 1

        self.action_space = spaces.Discrete(len(self.action_map))

        # --- Observation Space Definition ---
        # State includes: status of controllable lines + served status of load zones
        self._load_zone_nodes = sorted([n for n, data in self.nx_graph.nodes(data=True) if data['type'] == 'Load Zone'])
        self.state_length = len(self.controllable_lines_keys) + len(self._load_zone_nodes)
        # Observation space: Binary vector (0 or 1) for each element in the state
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_length,), dtype=np.int32)

        # --- Q-Learning Parameters ---
        self.alpha = 0.1      # Learning rate
        self.gamma = 0.99     # Discount factor
        self.epsilon = 1.0    # Initial exploration rate
        self.epsilon_decay = 0.995 # Rate at which epsilon decreases
        self.epsilon_min = 0.01   # Minimum exploration rate

        # Q-table and state mapping (will be initialized/loaded by main script)
        self.q_table = None
        self.state_to_index = None
        self.next_state_index = None
        self.current_q_table_capacity = 0

        # --- Internal State Caching ---
        self._last_blackout_status = None # Cache simulation results within a step

    def initialize_q_table(self, loaded_q_table=None, loaded_state_map=None, loaded_next_idx=None):
         """
         Initializes or loads the Q-table and state mapping.

         Args:
             loaded_q_table (np.ndarray, optional): Pre-loaded Q-table.
             loaded_state_map (dict, optional): Pre-loaded state-to-index map.
             loaded_next_idx (int, optional): Pre-loaded next available state index.
         """
         if loaded_q_table is not None and loaded_state_map is not None and loaded_next_idx is not None:
             self.q_table = loaded_q_table
             self.state_to_index = loaded_state_map
             self.next_state_index = loaded_next_idx
             self.current_q_table_capacity = self.q_table.shape[0]
             print(f"Env: Initialized with loaded Q-table (capacity: {self.current_q_table_capacity}, states: {self.next_state_index}).")
         else:
             # Initialize a new Q-table with default size
             self.q_table = np.zeros((self.initial_q_table_size, self.action_space.n), dtype=np.float32)
             self.state_to_index = {}
             self.next_state_index = 0
             self.current_q_table_capacity = self.initial_q_table_size
             print(f"Env: Initialized new Q-table (capacity: {self.current_q_table_capacity}).")

    def _state_to_index_lookup(self, state_tuple):
        """
        Gets the index for a given state tuple in the Q-table.
        Adds the state and expands the Q-table if it's a new state.

        Args:
            state_tuple (tuple): The state representation.

        Returns:
            int: The row index in the Q-table corresponding to the state.
        """
        if state_tuple in self.state_to_index:
            return self.state_to_index[state_tuple]
        else:
            # --- Expand Q-table if necessary ---
            if self.next_state_index >= self.current_q_table_capacity:
                new_capacity = self.current_q_table_capacity * 2
                print(f"Env: Resizing Q-table from {self.current_q_table_capacity} to {new_capacity}")
                # Pad rows with zeros, keeping the number of columns (actions) the same
                self.q_table = np.pad(self.q_table,
                                      ((0, self.current_q_table_capacity), (0, 0)),
                                      mode='constant', constant_values=0)
                self.current_q_table_capacity = new_capacity

            # --- Add new state ---
            new_index = self.next_state_index
            self.state_to_index[state_tuple] = new_index
            self.next_state_index += 1
            return new_index

    def _get_state_representation(self):
        """
        Generates the state tuple based on the current grid status.

        State includes:
        - Binary status (1=in service, 0=out of service) for each controllable line.
        - Binary status (1=served, 0=blacked out) for each load zone.

        Returns:
            tuple: The state representation.
        """
        state_list = []

        # 1. Controllable line statuses (ensure consistent order)
        for u, v in self.controllable_lines_keys:
            # Check if edge exists before accessing attributes (should always exist based on init)
            if self.current_graph.has_edge(u, v):
                 state_list.append(int(self.current_graph[u][v].get('in_service', False)))
            else:
                 # This case should ideally not happen if graph structure is static within episode
                 print(f"Warning: Controllable line ({u}, {v}) not found in current graph during state generation.")
                 state_list.append(0) # Assume out of service if missing

        # 2. Load zone served statuses (ensure consistent order)
        # Use cached blackout status if available from the last simulation within the step
        if self._last_blackout_status is None:
             # If cache is invalid (e.g., start of step, after reset), simulate to get status
             _, blackout_status, _ = simulate_grid_state(self.current_graph)
             self._last_blackout_status = blackout_status # Cache the result

        # Append status for each load zone (1 if served, 0 if blacked out)
        for load_node in self._load_zone_nodes: # Use the sorted list for consistency
            # Default to True (blacked out) if load_node somehow not in status dict
            is_blacked_out = self._last_blackout_status.get(load_node, True)
            state_list.append(int(not is_blacked_out))

        return tuple(state_list)

    def _calculate_reward(self, served_load_mw, blackout_status, overloaded_lines, action_cost):
        """
        Calculates the reward based on the outcome of the last step.

        Args:
            served_load_mw (float): Total load served.
            blackout_status (dict): Blackout status of load zones.
            overloaded_lines (set): Set of overloaded line tuples (u, v).
            action_cost (float): Cost associated with the action taken.

        Returns:
            float: The calculated reward value.
        """
        reward = 0
        total_initial_load = sum(self._initial_load_demands.values())

        # --- Positive Reward Component ---
        # Reward for served load (normalized by total possible load)
        if total_initial_load > 0:
            reward += 100 * (served_load_mw / total_initial_load)
        else:
            # Small reward if grid is stable but there's no load demand
            reward += 50

        # --- Negative Reward Components (Penalties) ---
        critical_blackout_penalty = 0
        non_critical_blackout_penalty = 0
        for load_node, is_blacked_out in blackout_status.items():
            if is_blacked_out:
                # Safely get node data, default 'critical' to False if node somehow missing
                node_data = self.current_graph.nodes.get(load_node, {})
                if node_data.get('critical', False):
                    critical_blackout_penalty += 1 # Count critical blackouts
                else:
                    non_critical_blackout_penalty += 1 # Count non-critical blackouts

        # Apply penalties (higher penalty for critical loads)
        reward -= critical_blackout_penalty * 100 # Heavier penalty
        reward -= non_critical_blackout_penalty * 20 # Lighter penalty

        # Penalty for each overloaded line
        reward -= 50 * len(overloaded_lines)

        # Cost for taking the action (e.g., cost for switching, higher cost for shedding)
        reward -= action_cost

        return reward

    def _update_q_table(self, state_tuple, action_idx, reward, next_state_tuple, done):
        """
        Updates the Q-table using the Bellman equation (Q-learning update rule).

        Args:
            state_tuple (tuple): The state before the action.
            action_idx (int): The index of the action taken.
            reward (float): The reward received.
            next_state_tuple (tuple): The state after the action.
            done (bool): Whether the episode terminated.
        """
        # Get Q-table indices for the current and next states
        state_idx = self._state_to_index_lookup(state_tuple)
        next_state_idx = self._state_to_index_lookup(next_state_tuple)

        # Get the current Q-value for the state-action pair
        current_q = self.q_table[state_idx, action_idx]

        # Find the maximum Q-value for the *next* state (best possible future reward)
        # If the episode is done, the future reward is 0
        max_next_q = np.max(self.q_table[next_state_idx]) if not done else 0.0

        # Q-learning formula: Q(s,a) = Q(s,a) + alpha * [R + gamma * max Q(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Update the Q-table
        self.q_table[state_idx, action_idx] = new_q

    def choose_action(self, state_representation):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state_representation (tuple): The current state tuple.

        Returns:
            int: The index of the chosen action.
        """
        # Exploration vs. Exploitation
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return self.action_space.sample()
        else:
            # Exploit: choose the best action based on current Q-values
            state_idx = self._state_to_index_lookup(state_representation)
            state_q_values = self.q_table[state_idx]

            # Find the action(s) with the highest Q-value
            # Use np.where to handle potential ties randomly
            best_action_indices = np.where(state_q_values == np.max(state_q_values))[0]
            return np.random.choice(best_action_indices)

    def step(self, action_index):
        """
        Executes one time step in the environment.

        Args:
            action_index (int): The index of the action selected by the agent.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation (tuple): The state representation after the step.
                - reward (float): The reward received for the step.
                - terminated (bool): Whether the episode ended (e.g., critical failure).
                - truncated (bool): Whether the episode was cut short (e.g., max steps).
                - info (dict): Auxiliary information about the step.
        """
        # --- Get state BEFORE applying action and dynamics ---
        current_state_tuple = self._get_state_representation() # Uses cached blackout status if available

        # --- 1. Auto-Healing ---
        healed_lines_info = [] # Store info about lines that were healed
        if self.auto_heal_rate > 0:
             # Iterate over a copy of edge keys, as graph might change if healing occurs
             edges_to_check = list(self.current_graph.edges())
             for u, v in edges_to_check:
                 # Check if it's a controllable line and currently out of service
                 if (u, v) in self.controllable_lines_set and \
                    self.current_graph.has_edge(u,v) and \
                    not self.current_graph[u][v].get('in_service', True):
                     # Probabilistic healing
                     if random.random() < self.auto_heal_rate:
                         self.current_graph[u][v]['in_service'] = True
                         healed_lines_info.append((u, v))
                         # If the reverse edge is also controllable, assume it heals too (common model)
                         if (v, u) in self.controllable_lines_set and self.current_graph.has_edge(v, u):
                              if not self.current_graph[v][u].get('in_service', True):
                                  self.current_graph[v][u]['in_service'] = True
                                  # Don't double add to healed_lines_info, handled by reporting logic

        # --- 2. Apply Agent's Action ---
        action_type, action_arg = self.action_map[action_index]
        action_cost = 0.1 # Default small cost for line switching/noop
        shed_amount_mw = 0 # Track how much load was shed

        if action_type == 'noop':
            action_cost = 0
        elif action_type == 'open_line':
            u, v = action_arg
            if self.current_graph.has_edge(u, v):
                self.current_graph[u][v]['in_service'] = False
                # If the reverse edge is controllable, open it too for consistency
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
                shed_amount_mw = current_demand * self.shed_percentage
                # Reduce demand, ensuring it doesn't go below zero
                self.current_graph.nodes[load_node]['current_demand_mw'] = max(0, current_demand - shed_amount_mw)
                # Action cost proportional to the amount of load shed (tune factor 0.5)
                action_cost = shed_amount_mw * 0.5

        # --- 3. Fault Simulation ---
        faulted_line_key = None # Store info about line that faulted
        # Get controllable lines that are currently IN SERVICE (potential candidates for fault)
        in_service_controllable_lines = [(u, v) for u, v in self.controllable_lines_keys
                                          if self.current_graph.has_edge(u, v) and self.current_graph[u][v].get('in_service', False)]

        # Probabilistic fault occurrence
        if in_service_controllable_lines and random.random() < self.fault_rate:
            # Choose a random in-service controllable line to fault
            faulted_line_key = random.choice(in_service_controllable_lines)
            u_fault, v_fault = faulted_line_key
            # Take the faulted line out of service
            if self.current_graph.has_edge(u_fault, v_fault):
                 self.current_graph[u_fault][v_fault]['in_service'] = False
            # Take the reverse direction out too if it's controllable and exists
            if (v_fault, u_fault) in self.controllable_lines_set and self.current_graph.has_edge(v_fault, u_fault):
                self.current_graph[v_fault][u_fault]['in_service'] = False
            self.steps_since_last_fault = 0 # Reset counter
        else:
            self.steps_since_last_fault += 1

        # --- 4. Simulate Grid State & Get Next State ---
        # CRUCIAL: Invalidate the blackout cache before re-simulating
        self._last_blackout_status = None
        served_load_mw, blackout_status, overloaded_lines = simulate_grid_state(self.current_graph)
        # Update cache AFTER simulation for potential use in reward calc or next state gen
        self._last_blackout_status = blackout_status
        # Get the state representation AFTER all dynamics and simulation
        next_state_tuple = self._get_state_representation()

        # --- 5. Calculate Reward & Termination Conditions ---
        reward = self._calculate_reward(served_load_mw, blackout_status, overloaded_lines, action_cost)

        # Define 'terminated' condition (e.g., simulation ends if all critical loads are blacked out)
        critical_nodes = [n for n, data in self.current_graph.nodes(data=True)
                           if data.get('critical', False) and data['type'] == 'Load Zone']
        all_critical_blacked_out = False
        if critical_nodes: # Only check if critical nodes actually exist
             # Check if all nodes in the critical_nodes list are blacked out
             all_critical_blacked_out = all(blackout_status.get(n, True) for n in critical_nodes)
        terminated = all_critical_blacked_out

        # 'truncated' is usually based on time limits, handled by the training loop (max_steps)
        truncated = False # Set by the training loop if max_steps is reached

        # --- 6. Q-Table Update (if learning is enabled, alpha > 0) ---
        if self.alpha > 0:
             self._update_q_table(current_state_tuple, action_index, reward, next_state_tuple, terminated)

        # --- 7. Info Dictionary ---
        info = {
            'served_load_mw': served_load_mw,
            'blackout_status': blackout_status,
            'overloaded_lines': list(overloaded_lines), # Convert set to list for JSON compatibility if needed
            'faulted_line': faulted_line_key,
            'healed_lines': healed_lines_info,
            'action_taken': self.action_map[action_index], # Include details of the action performed
            'action_cost': action_cost,
            'load_shed_mw': shed_amount_mw
        }

        # --- Return according to Gymnasium API ---
        return next_state_tuple, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new episode.

        Args:
            seed (int, optional): Seed for the random number generator.
            options (dict, optional): Additional options for resetting.

        Returns:
            tuple: (observation, info)
                - observation (tuple): The initial state representation.
                - info (dict): Auxiliary information about the reset.
        """
        super().reset(seed=seed) # Handles seeding for internal RNGs if needed

        # Create a fresh copy of the base graph for the new episode
        self.current_graph = self.nx_graph.copy()

        # Reset edge states to their initial configuration
        for (u, v), initial_status in self._initial_edge_states.items():
            if self.current_graph.has_edge(u, v):
                 self.current_graph[u][v]['in_service'] = initial_status
            # Also reset reverse edge if it exists and mirrors the state (common in models)
            # Check if the reverse edge key exists in the initial states dictionary
            if self.current_graph.has_edge(v, u) and (v, u) in self._initial_edge_states:
                 self.current_graph[v][u]['in_service'] = self._initial_edge_states[(v, u)]
            # Handle cases where only one direction might be in initial_edge_states explicitly
            elif self.current_graph.has_edge(v, u) and (u, v) in self._initial_edge_states:
                 # If reverse isn't defined, maybe assume it mirrors the forward state? Or handle based on model needs.
                 # For safety, only reset if explicitly defined or assumed symmetric.
                 # Here, we assume symmetry if the reverse key isn't present but forward is.
                 # self.current_graph[v][u]['in_service'] = initial_status # Uncomment if symmetry is assumed
                 pass # Or do nothing if reverse state isn't explicitly defined

        # Reset load demands to their initial values
        for node, initial_demand in self._initial_load_demands.items():
            if node in self.current_graph.nodes():
                 # Ensure 'current_demand_mw' attribute exists before setting
                 if 'current_demand_mw' in self.current_graph.nodes[node]:
                     self.current_graph.nodes[node]['current_demand_mw'] = initial_demand
                 else:
                     # This might happen if load nodes were defined without 'current_demand_mw' initially
                     print(f"Warning: 'current_demand_mw' attribute missing for node {node} during reset.")


        # Reset episode-specific variables
        self.steps_since_last_fault = 0
        self._last_blackout_status = None # Clear internal cache

        # Get the initial state representation of the reset environment
        initial_state_tuple = self._get_state_representation()

        # Decay epsilon at the start of each new episode (during training)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Standard Gym API return: observation, info dictionary
        info = {} # No extra info needed on reset by default
        return initial_state_tuple, info

    def render(self):
        """
        Rendering is handled by the main script in this modular setup.
        This method is kept for API compatibility but does nothing.
        """
        # Rendering logic is moved to main_script.py or a dedicated visualization module
        # print("Rendering is handled by the main script.")
        pass

    def close(self):
        """
        Clean up any resources (like plot windows) associated with the environment.
        In this setup, closing is handled by the main script.
        """
        # Closing logic (e.g., closing plot windows) is moved to main_script.py
        # print("Closing resources is handled by the main script.")
        pass

# Example usage (optional, for testing this module)
if __name__ == '__main__':
    from grid_topology import create_grid_topology
    from grid_utils import load_q_table # Need this for initialization if testing loading

    # 1. Create grid components
    graph, init_edges, init_loads, _ = create_grid_topology()

    # 2. Initialize Environment
    env = SimplePowerGridEnv(graph, init_edges, init_loads, fault_rate=0.1, auto_heal_rate=0.05)

    # 3. Initialize Q-table (example: load or create new)
    action_size = env.action_space.n
    state_len = env.state_length
    q_table, state_map, next_idx = load_q_table('dummy_q_table.npz', action_size, state_len) # Try loading
    env.initialize_q_table(q_table, state_map, next_idx) # Pass loaded/new table to env

    # 4. Reset environment
    print("Resetting environment...")
    state, info = env.reset()
    print(f"Initial State: {state}")
    print(f"Initial Info: {info}")

    # 5. Run a few random steps
    print("\nRunning a few random steps...")
    for i in range(5):
        action = env.action_space.sample() # Choose random action
        print(f"--- Step {i+1} ---")
        print(f"Action Taken: {env.action_map[action]}")
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Next State: {next_state}")
        print(f"Reward: {reward:.2f}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        state = next_state
        if terminated or truncated:
            print("Episode ended.")
            break

    print("\nEnvironment test complete.")
