import os
import random
import argparse # for running the script
import collections # Added for defaultdict
# defaultdict: automatically initializes missing keys with a default value

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from conf import *

# --- Grid Definition ---

def define_grid_topology():
    """Defines the grid network structure."""
    G = nx.Graph()

    # Add Nodes: Generators, Substations, Load Zones
    G.add_node("G1", type=GENERATOR)
    G.add_node("G2", type=GENERATOR)

    G.add_node("S1", type=SUBSTATION)
    G.add_node("S2", type=SUBSTATION)
    G.add_node("S3", type=SUBSTATION)
    G.add_node("S4", type=SUBSTATION)

    G.add_node("L1", type=LOAD_ZONE, critical=False)
    G.add_node("L2", type=LOAD_ZONE, critical=True) # Critical Load 1
    G.add_node("L3", type=LOAD_ZONE, critical=False)
    G.add_node("L4", type=LOAD_ZONE, critical=True) # Critical Load 2
    G.add_node("L5", type=LOAD_ZONE, critical=False)

    # Add Connections (Edges) with initial status and controllability
    # Define edges as (node1, node2, {'status': initial_status, 'controllable': bool})
    edges = [
        ("G1", "S1", {'controllable': False}),
        ("G2", "S2", {'controllable': False}),

        ("S1", "S2", {'controllable': True}),
        ("S1", "S3", {'controllable': True}),
        ("S2", "S4", {'controllable': True}),
        ("S3", "S4", {'controllable': True}), # Redundant path
        ("S1", "S4", {'controllable': True}), # Another path

        ("S3", "L1", {'controllable': False}),
        ("S3", "L2", {'controllable': False}),
        ("S4", "L3", {'controllable': False}),
        ("S4", "L4", {'controllable': False}),
        ("S2", "L5", {'controllable': False}),
        ("S3", "L5", {'controllable': False}), # L5 has dual supply potential
    ]
    
    # add all edjes to be IN_SERVICE so we don't need to add it manually since they all will be IN_SERVICE
    for u, v, attrs in edges:
        attrs['status'] = IN_SERVICE
        G.add_edge(u, v, **attrs)

    return G

# --- Simulation Logic ---

def get_powered_status(graph):
    """
    Determines which load zones are powered based on connectivity to generators.

    Args:
        graph (nx.Graph): The current grid graph with edge statuses.

    Returns:
        dict: A dictionary mapping load zone node IDs to boolean (True if powered).
    """
    powered_status = {}
    generators = {n for n, d in graph.nodes(data=True) if d['type'] == GENERATOR}
    load_zones = {n for n, d in graph.nodes(data=True) if d['type'] == LOAD_ZONE}

    in_service_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['status'] == IN_SERVICE]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(graph.nodes()) # Add all nodes, even if isolated
    subgraph.add_edges_from(in_service_edges)

    for lz in load_zones:
        is_powered = False
        for gen in generators:
            if nx.has_path(subgraph, source=lz, target=gen):
                is_powered = True
                break
        powered_status[lz] = is_powered

    return powered_status

# --- Reinforcement Learning Environment ---

class PowerGridEnv(gym.Env):
    """
    Custom Gym environment for the simplified power grid simulation.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, graph):
        super().__init__()

        self.initial_graph = graph.copy()
        self.graph = None
        self.current_step = 0

        self.generators = {n for n, d in self.initial_graph.nodes(data=True) if d['type'] == GENERATOR}
        self.load_zones = {n: d for n, d in self.initial_graph.nodes(data=True) if d['type'] == LOAD_ZONE}
        self.substations = {n for n, d in self.initial_graph.nodes(data=True) if d['type'] == SUBSTATION}

        # Store edges in a fixed order for consistent observation/action mapping
        self.edge_list = list(self.initial_graph.edges(data=True))
        self.controllable_edges_indices = [
            i for i, (u, v, d) in enumerate(self.edge_list) if d.get('controllable', False)
        ]
        self.num_controllable_edges = len(self.controllable_edges_indices)

        # Define Action Space: Discrete action for each controllable edge + 'do nothing'
        # Action 0: Do nothing
        # Action i+1: Toggle status of controllable_edge[i]
        self.action_space = spaces.Discrete(self.num_controllable_edges + 1)

        # Define Observation Space: Binary status of ALL edges (1=in_service, 0=out_of_service)
        # For SARSA with Q-table, we'll convert this to a hashable tuple
        self.observation_space = spaces.MultiBinary(len(self.edge_list))

        # For visualization
        self.fig = None
        self.ax = None
        self.pos = None

    def _get_observation(self):
        """Returns the current observation as a hashable tuple."""
        obs = tuple(1 if d['status'] == IN_SERVICE else 0 for u, v, d in self.graph.edges(data=True))
        return obs

    def _apply_action(self, action):
        """Applies the chosen action to the grid graph."""
        action_penalty = 0.0
        if action == 0: # Do nothing
            pass
        elif 1 <= action <= self.num_controllable_edges:
            edge_index_in_list = self.controllable_edges_indices[action - 1]
            # Get edge endpoints from the fixed list to index into the current graph
            u, v, _ = self.edge_list[edge_index_in_list]

            # Important: Update the *current* graph state
            current_status = self.graph.edges[u, v]['status']
            new_status = OUT_OF_SERVICE if current_status == IN_SERVICE else IN_SERVICE
            self.graph.edges[u, v]['status'] = new_status

            action_penalty = PENALTY_SWITCHING_ACTION
        else:
            print(f"Warning: Invalid action {action} received.")

        return action_penalty

    def _apply_random_events(self):
        """Simulates random line failures and repairs."""
        for u, v, data in self.graph.edges(data=True):
            status = data['status']
            if status == IN_SERVICE:
                if random.random() < PROB_LINE_FAILURE:
                    self.graph.edges[u, v]['status'] = OUT_OF_SERVICE
            else: # status == OUT_OF_SERVICE
                if random.random() < PROB_LINE_REPAIR:
                   self.graph.edges[u, v]['status'] = IN_SERVICE

    def _calculate_reward(self, powered_status):
        """Calculates the reward based on the current grid state."""
        reward = 0.0
        for lz, data in self.load_zones.items():
            is_critical = data.get('critical', False)
            is_powered = powered_status.get(lz, False)

            if is_powered:
                reward += REWARD_CRITICAL_LOAD_POWERED if is_critical else REWARD_LOAD_POWERED
            else:
                reward += PENALTY_CRITICAL_LOAD_BLACKOUT if is_critical else PENALTY_LOAD_BLACKOUT
        return reward

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self.graph = self.initial_graph.copy()
        # Optional: Apply some initial random outages
        # self._apply_random_events()

        self.current_step = 0
        observation = self._get_observation()
        info = self._get_info()

        if self.fig is not None:
             plt.close(self.fig)
             self.fig = None
             self.ax = None
             self.pos = None # Recalculate layout on reset

        return observation, info

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1

        action_penalty = self._apply_action(action)
        self._apply_random_events()
        powered_status = get_powered_status(self.graph)

        reward = self._calculate_reward(powered_status) + action_penalty

        terminated = self.current_step >= MAX_STEPS_PER_EPISODE
        truncated = False

        observation = self._get_observation()
        info = self._get_info(powered_status)

        return observation, reward, terminated, truncated, info

    def _get_info(self, powered_status=None):
        """Returns auxiliary information (optional)."""
        if powered_status is None:
             powered_status = get_powered_status(self.graph)
        return {
            "step": self.current_step,
            "powered_status": powered_status,
            "num_powered_loads": sum(powered_status.values()),
            "num_critical_powered": sum(1 for lz, p in powered_status.items() if self.load_zones[lz].get('critical') and p)
        }

    def render(self, mode='human'):
        """Renders the current grid state using Matplotlib."""
        if mode != 'human':
            return

        if self.fig is None or self.pos is None:
            plt.ion() # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            # Compute layout once
            self.pos = nx.spring_layout(self.graph, seed=42) # Use a fixed seed for consistent layout

        self.ax.clear()

        # Define colors
        node_colors = []
        node_labels = {}
        powered_status = get_powered_status(self.graph)

        for node, data in self.graph.nodes(data=True):
            node_labels[node] = node # Simple label
            node_type = data['type']
            if node_type == GENERATOR:
                node_colors.append('green')
            elif node_type == SUBSTATION:
                node_colors.append('blue')
            elif node_type == LOAD_ZONE:
                is_critical = data.get('critical', False)
                is_powered = powered_status.get(node, False)
                if is_powered:
                    node_colors.append('yellow' if not is_critical else 'orange') # Powered
                else:
                    node_colors.append('grey' if not is_critical else 'darkred') # Blackout
            else:
                node_colors.append('grey')

        # Edge colors based on status and controllability for visualization clarity
        edge_colors = []
        edge_widths = []
        edge_styles = []
        for u, v, data in self.graph.edges(data=True):
            status = data['status']
            controllable = data.get('controllable', False)

            if status == IN_SERVICE:
                edge_colors.append('green')
                edge_widths.append(2.5)
            else: # OUT_OF_SERVICE
                edge_colors.append('red')
                edge_widths.append(1.0)

            edge_styles.append('solid' if controllable else 'dashed')


        # Need to draw edges explicitly to control styles per edge
        # Draw all edges first
        nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax, edge_color=edge_colors, width=edge_widths, style=edge_styles)
        # Draw nodes on top
        nx.draw_networkx_nodes(self.graph, self.pos, ax=self.ax, node_color=node_colors, node_size=600)
        # Add labels on top of nodes
        nx.draw_networkx_labels(self.graph, self.pos, labels=node_labels, ax=self.ax, font_size=10)


        # Add a title with step info
        powered_count = sum(1 for p in powered_status.values() if p)
        critical_powered = sum(1 for lz, p in powered_status.items() if self.load_zones[lz].get('critical') and p)
        critical_total = sum(1 for lz, d in self.load_zones.items() if d.get('critical'))
        self.ax.set_title(f"Step: {self.current_step} | Powered Loads: {powered_count}/{len(self.load_zones)} | Critical Powered: {critical_powered}/{critical_total}")

        plt.draw()
        plt.pause(VISUALIZATION_PAUSE)

    def close(self):
        """Closes the rendering window."""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff() # Turn off interactive mode
            self.fig = None
            self.ax = None
            self.pos = None


# --- SARSA Agent ---

class SarsaAgent:
    def __init__(self, action_space_n, learning_rate, discount_factor, exploration_rate):
        self.action_space_n = action_space_n
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Q-table: state (tuple) -> action (int) -> Q-value (float)
        # Using defaultdict makes it easy to handle unseen states, initializing their Q-values to zeros.
        self.q_table = collections.defaultdict(lambda: np.zeros(self.action_space_n))

    def choose_action(self, state, explore=True):
        """Chooses an action using an epsilon-greedy policy."""
        if explore and random.random() < self.exploration_rate:
            # Explore: Choose a random action
            return random.randrange(self.action_space_n)
        else:
            # Exploit: Choose the action with the highest Q-value
            q_values = self.q_table[state]
            # Handle tie-breaking randomly
            max_q = np.max(q_values)
            # Get indices of actions with the max Q-value
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions) # Pick one randomly

    def learn(self, state, action, reward, next_state, next_action):
        """Performs the SARSA update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]."""
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action] # Q(s', a')

        td_target = reward + self.discount_factor * next_q
        td_error = td_target - current_q

        self.q_table[state][action] += self.learning_rate * td_error

    def save(self, filepath):
         """Saves the Q-table using numpy."""
         # Convert defaultdict to dict before saving
         data_to_save = dict(self.q_table)
         np.save(filepath, data_to_save)
         print(f"SARSA Q-table saved to {filepath}")

    def load(self, filepath):
         """Loads the Q-table using numpy."""
         try:
             # Load the numpy file, access the dictionary within
             loaded_data = np.load(filepath, allow_pickle=True).item()
             # Recreate the defaultdict from the loaded dictionary
             self.q_table = collections.defaultdict(lambda: np.zeros(self.action_space_n), loaded_data)
             print(f"SARSA Q-table loaded from {filepath}")
             return True # Indicate successful load
         except FileNotFoundError:
             print(f"Error: Q-table file not found at {filepath}")
             return False
         except Exception as e:
             print(f"Error loading Q-table: {e}")
             return False


# --- Training Function (SARSA) ---

def train_sarsa_agent(env, agent, num_episodes=1000):
    """Trains the SARSA agent over multiple episodes."""
    print(f"\n--- Starting SARSA Training for {num_episodes} episodes ---")
    for episode in range(num_episodes):
        state, info = env.reset()
        # Choose the first action based on the initial state using the epsilon-greedy policy
        action = agent.choose_action(state, explore=True)
        total_reward = 0

        # Use a progress indicator
        if (episode + 1) % (num_episodes // 10 or 1) == 0:
             print(f"Episode {episode + 1}/{num_episodes}")

        for step in range(MAX_STEPS_PER_EPISODE):
            # Execute the chosen action
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                # At a terminal state, the SARSA update doesn't involve Q(s', a') because there's no next state/action
                # The reward is the final reward. The target is just the reward.
                # Q(s, a) <- Q(s, a) + alpha * [r - Q(s, a)]
                agent.learn(state, action, reward, state, 0) # Use current state and dummy next action for learning
                break
            else:
                 # Choose the *next* action (a') from the next state (s') using the current policy
                next_action = agent.choose_action(next_state, explore=True)
                 # Perform the SARSA update: Q(s, a) using r, s', and a'
                agent.learn(state, action, reward, next_state, next_action)

                 # Update state and action for the next step in the episode
                state = next_state
                action = next_action

        # Optional: Log reward per episode
        # print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")

    print("--- SARSA Training Complete ---")


# --- Demonstration Function (SARSA) ---

def run_sarsa_simulation(env, agent, num_steps=100):
    """Runs the simulation with the trained SARSA agent and visualizes it."""
    print("\n--- Starting SARSA Demonstration ---")
    state, info = env.reset()
    total_reward = 0
    env.render(mode='human') # Initial state

    for step in range(num_steps):
        # Choose the action greedily (explore=False)
        action = agent.choose_action(state, explore=False)
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        print(f"Step: {info['step']}, Action: {action}, Reward: {reward:.2f}, Powered: {info['num_powered_loads']}, Crit Powered: {info['num_critical_powered']}")

        env.render(mode='human')
        state = next_state # Update state for the next step

        if terminated or truncated:
            print("Episode finished.")
            break

    print(f"--- Demonstration Complete --- Total Reward: {total_reward:.2f}")
    env.close()


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Power Grid and Train/Run SARSA Controller")
    parser.add_argument("--mode", choices=['train', 'demo'], default='train', help="Run in training or demonstration mode")
    parser.add_argument("--load", action='store_true', help="Load pre-trained agent knowledge")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes for SARSA training")
    parser.add_argument("--demosteps", type=int, default=50, help="Number of steps for demonstration")
    parser.add_argument("--qtablepath", type=str, default=MODEL_FILENAME, help="Path to save/load the Q-table (.npy file)")

    args = parser.parse_args()

    # Define the grid
    grid_topology = define_grid_topology()

    # Create the environment
    env = PowerGridEnv(grid_topology)

    # Create the SARSA agent
    sarsa_agent = SarsaAgent(
        action_space_n=env.action_space.n,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        exploration_rate=EXPLORATION_RATE
    )

    # Load agent knowledge if requested and file exists
    if args.load:
        sarsa_agent.load(args.qtablepath)

    if args.mode == 'train':
        # If loading for train and load was successful, continue training.
        # Otherwise, start training from scratch (defaultdict handles initial state values)
        train_sarsa_agent(env, sarsa_agent, num_episodes=args.episodes)
        sarsa_agent.save(args.qtablepath) # Save after training

    elif args.mode == 'demo':
        # In demo mode, loading is required
        if not args.load or not os.path.exists(args.qtablepath):
             print(f"Error: Model file not found at {args.qtablepath}. Train a model first using --mode train.")
             exit()
        run_sarsa_simulation(env, sarsa_agent, num_steps=args.demosteps)

    # Ensure environment resources are cleaned up if not running demo mode
    if args.mode != 'demo':
         env.close()