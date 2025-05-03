# -*- coding: utf-8 -*-
"""
Power Grid Control Simulation using Deep Reinforcement Learning (DQN) with Stable Baselines3

This project simulates a simplified power grid focused on maintaining connectivity.
It uses the DQN algorithm from the Stable Baselines3 library to train an agent
to open and close transmission lines/ties to prevent critical load zone blackouts
due to disconnections. Power flow simulation, overload handling, and numerical
demand/capacity calculations are excluded.

Dependencies:
- networkx: For graph representation and analysis.
- matplotlib: For visualization.
- numpy: For numerical operations.
- gymnasium: For the Reinforcement Learning environment structure.
- stable-baselines3[extra]: For DQN and necessary utilities.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import random
import time
import os
from networkx.algorithms.components import connected_components
import copy

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# --- Constants ---
MODEL_FILENAME = 'dqn_power_grid_connectivity_model'
LOG_DIR = './logs/'

# --- Grid Topology Definition ---
G = nx.DiGraph()
node_types = {
    'Generator': ['Generator 1 (G1)', 'Generator 2 (G2)', 'Generator 3 (G3)'],
    'Substation': ['Substation A', 'Substation B', 'Substation C', 'Substation D',
                   'Substation E', 'Substation F', 'Substation G', 'Substation H',
                   'Substation I', 'Substation J'],
    'Load Zone': ['Load Zone 1 (Critical)', 'Load Zone 2', 'Load Zone 3 (Critical)', 'Load Zone 4'],
}
for node_type, nodes in node_types.items():
    for node in nodes:
        G.add_node(node, type=node_type)
edges = [
    # Removed capacity_mw as it's not used in this connectivity model
    ('Generator 1 (G1)', 'Substation A', {'label': 'G1-SA', 'in_service': True, 'critical': True, 'component': 'transmission'}),
    ('Generator 2 (G2)', 'Substation F', {'label': 'G2-SF', 'in_service': True, 'critical': True, 'component': 'transmission'}),
    ('Generator 3 (G3)', 'Substation H', {'label': 'G3-SH', 'in_service': True, 'critical': False, 'component': 'transmission'}),
    ('Substation A', 'Substation B', {'label': 'SA-SB', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation B', 'Substation A', {'label': 'SB-SA', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation A', 'Substation C', {'label': 'SA-SC', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation C', 'Substation A', {'label': 'SC-SA', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation B', 'Substation D', {'label': 'SB-SD', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation D', 'Substation B', {'label': 'SD-SB', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation C', 'Substation E', {'label': 'SC-SE', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation E', 'Substation C', {'label': 'SE-SC', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation F', 'Substation G', {'label': 'SF-SG', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation G', 'Substation F', {'label': 'SG-SF', 'in_service': True, 'critical': True, 'component': 'line'}),
    ('Substation F', 'Substation I', {'label': 'SF-SI', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation I', 'Substation F', {'label': 'SI-SF', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation G', 'Substation J', {'label': 'SG-SJ', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation J', 'Substation G', {'label': 'SJ-SG', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation H', 'Substation I', {'label': 'SH-SI', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation I', 'Substation H', {'label': 'SI-SH', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation D', 'Substation G', {'label': 'SD-SG (Tie)', 'in_service': True, 'critical': False, 'component': 'tie_line'}),
    ('Substation G', 'Substation D', {'label': 'SG-SD (Tie)', 'in_service': True, 'critical': False, 'component': 'tie_line'}),
    ('Substation E', 'Substation J', {'label': 'SE-SJ (Tie)', 'in_service': True, 'critical': False, 'component': 'tie_line'}),
    ('Substation J', 'Substation E', {'label': 'SJ-SE (Tie)', 'in_service': True, 'critical': False, 'component': 'tie_line'}),
    ('Substation I', 'Substation J', {'label': 'SI-SJ', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation J', 'Substation I', {'label': 'SJ-SI', 'in_service': True, 'critical': False, 'component': 'line'}),
    ('Substation B', 'Load Zone 1 (Critical)', {'label': 'Feeder B-LZ1', 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation D', 'Load Zone 1 (Critical)', {'label': 'Feeder D-LZ1', 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation C', 'Load Zone 2', {'label': 'Feeder C-LZ2', 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation E', 'Load Zone 2', {'label': 'Feeder E-LZ2', 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation G', 'Load Zone 3 (Critical)', {'label': 'Feeder G-LZ3', 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation J', 'Load Zone 3 (Critical)', {'label': 'Feeder J-LZ3', 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation I', 'Load Zone 4', {'label': 'Feeder I-LZ4', 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ('Substation H', 'Load Zone 4', {'label': 'Feeder H-LZ4', 'in_service': True, 'critical': False, 'component': 'feeder'}),
]
G.add_edges_from(edges)
G.nodes['Load Zone 1 (Critical)']['critical'] = True
G.nodes['Load Zone 2']['critical'] = False
G.nodes['Load Zone 3 (Critical)']['critical'] = True
G.nodes['Load Zone 4']['critical'] = False
initial_edge_states = {(u, v): data['in_service'] for u, v, data in G.edges(data=True)}
pos = {
    'Generator 1 (G1)': (-4, 10), 'Substation A': (-2, 8), 'Substation B': (-4, 6), 'Substation C': (0, 6),
    'Substation D': (-4, 4), 'Substation E': (0, 4),
    'Generator 2 (G2)': (4, 10), 'Substation F': (2, 8), 'Substation G': (4, 6), 'Substation J': (2, 4),
    'Generator 3 (G3)': (8, 10), 'Substation H': (6, 8), 'Substation I': (6, 6),
    'Load Zone 1 (Critical)': (-5, 2), 'Load Zone 2': (0, 2),
    'Load Zone 3 (Critical)': (3, 2), 'Load Zone 4': (7, 2),
}
pos = {node: p for node, p in pos.items() if node in G.nodes()}

# --- SIMPLIFIED Simulation Logic (Connectivity Only) ---
def simulate_grid_state(graph):
    load_nodes = [n for n, data in graph.nodes(data=True) if data['type'] == 'Load Zone']
    n_loads = len(load_nodes)
    blackout_status = {n: True for n in load_nodes}
    if graph.number_of_nodes() == 0:
        return 0.0, blackout_status
    active_generators = [n for n, data in graph.nodes(data=True) if data['type'] == 'Generator']
    if not active_generators:
        return 0.0, blackout_status
    operating_backbone_edges = [(u, v) for u, v, data in graph.edges(data=True)
                                if data.get('in_service', True) and
                                data.get('component') in ('transmission', 'tie_line', 'line')]
    operating_graph_connectivity = nx.Graph()
    nodes_in_backbone = active_generators + [n for n, data in graph.nodes(data=True) if data['type'] == 'Substation']
    operating_graph_connectivity.add_nodes_from(nodes_in_backbone)
    operating_graph_connectivity.add_edges_from(operating_backbone_edges)
    gen_connected_component_nodes = set()
    components = list(connected_components(operating_graph_connectivity))
    for comp in components:
        if any(node in active_generators for node in comp):
            gen_connected_component_nodes.update(comp)
    for load_node in load_nodes:
        feeder_sources = [u for u, v, data in graph.edges(data=True)
                          if v == load_node and
                          data.get('in_service', True) and
                          data.get('component') == 'feeder']
        if any(source_node in gen_connected_component_nodes for source_node in feeder_sources):
            blackout_status[load_node] = False
    blackout_flags_np = np.array(list(blackout_status.values()), dtype=bool) # Get flags in consistent order
    connected_load_count = count_connected_loads(blackout_flags_np, n_loads)
    return connected_load_count, blackout_status

# --- RL Environment Definition (Gymnasium Compatible) ---
class SimplePowerGridEnv(gym.Env):
    def __init__(self, nx_graph, initial_edge_states, fault_rate=0.05, auto_heal_rate=0.02):
        super().__init__()
        self.nx_graph = nx_graph
        self._initial_edge_states = initial_edge_states
        self.fault_rate = fault_rate
        self.auto_heal_rate = auto_heal_rate
        self.current_graph = copy.deepcopy(self.nx_graph)
        self.controllable_lines_keys = sorted([
            (u, v) for u, v, data in self.nx_graph.edges(data=True)
            if data.get('component') in ['transmission', 'tie_line', 'line']
        ])
        self.controllable_lines_set = set(self.controllable_lines_keys)

        self.action_map = {0: ('noop', None)}
        action_idx = 1
        for u, v in self.controllable_lines_keys:
            self.action_map[action_idx] = ('open_line', (u, v)); action_idx += 1
            self.action_map[action_idx] = ('close_line', (u, v)); action_idx += 1
        self.action_space = spaces.Discrete(len(self.action_map))

        self._load_zone_nodes = sorted([
            n for n, data in self.nx_graph.nodes(data=True) if data['type'] == 'Load Zone'
        ])
        self.state_length = len(self.controllable_lines_keys) + len(self._load_zone_nodes)
        self.observation_space = spaces.MultiBinary(self.state_length)

        self.steps_since_last_fault = 0
        self._last_blackout_status = None
        self.fig = None
        self.ax = None
        self.pos = pos

    def _get_state_representation(self):
        state_list = []
        for u, v in self.controllable_lines_keys:
            status = 0
            if self.current_graph.has_edge(u, v):
                status = int(self.current_graph.edges[u, v].get('in_service', False))
            state_list.append(status)
        if self._last_blackout_status is None:
            _, blackout_status = simulate_grid_state(self.current_graph)
            self._last_blackout_status = blackout_status
        for load_node in self._load_zone_nodes:
            is_blacked_out = self._last_blackout_status.get(load_node, True)
            state_list.append(int(not is_blacked_out))
        return np.array(state_list, dtype=np.int32)

    def _calculate_reward(self, blackout_status, action_cost):
        reward = 0.0
        powered_critical_reward = 150.0
        powered_normal_reward = 30.0
        blackout_critical_penalty = -300.0
        blackout_normal_penalty = -75.0
        for load_node, is_blacked_out in blackout_status.items():
            node_data = self.current_graph.nodes.get(load_node, {})
            is_critical = node_data.get('critical', False)
            if is_blacked_out:
                reward += blackout_critical_penalty if is_critical else blackout_normal_penalty
            else:
                reward += powered_critical_reward if is_critical else powered_normal_reward
        reward -= action_cost
        return reward

    def step(self, action_index):
        healed_lines_info = []
        if self.auto_heal_rate > 0:
             for u, v in self.controllable_lines_keys:
                 if self.current_graph.has_edge(u, v) and not self.current_graph.edges[u, v].get('in_service', True):
                     if random.random() < self.auto_heal_rate:
                         self.current_graph.edges[u, v]['in_service'] = True
                         healed_lines_info.append((u, v))
                         if (v, u) in self.controllable_lines_set and self.current_graph.has_edge(v, u):
                            if not self.current_graph.edges[v, u].get('in_service', True):
                                self.current_graph.edges[v, u]['in_service'] = True
        action_type, action_arg = self.action_map[action_index]
        action_cost = 0.0
        line_switched = False
        if action_type == 'open_line':
            u, v = action_arg
            if self.current_graph.has_edge(u, v) and self.current_graph.edges[u, v].get('in_service', False):
                self.current_graph.edges[u, v]['in_service'] = False
                line_switched = True
                if (v, u) in self.controllable_lines_set and self.current_graph.has_edge(v, u):
                    self.current_graph.edges[v, u]['in_service'] = False
        elif action_type == 'close_line':
            u, v = action_arg
            if self.current_graph.has_edge(u, v) and not self.current_graph.edges[u, v].get('in_service', True):
                self.current_graph.edges[u, v]['in_service'] = True
                line_switched = True
                if (v, u) in self.controllable_lines_set and self.current_graph.has_edge(v, u):
                    self.current_graph.edges[v, u]['in_service'] = True
        if line_switched:
            action_cost = 0.5
        faulted_line_key = None
        in_service_controllable_lines = [(u, v) for u, v in self.controllable_lines_keys
                                          if self.current_graph.has_edge(u, v) and self.current_graph.edges[u, v].get('in_service', False)]
        if in_service_controllable_lines and random.random() < self.fault_rate:
            faulted_line_key = random.choice(in_service_controllable_lines)
            u_fault, v_fault = faulted_line_key
            if self.current_graph.has_edge(u_fault, v_fault):
                 self.current_graph.edges[u_fault, v_fault]['in_service'] = False
            if (v_fault, u_fault) in self.controllable_lines_set and self.current_graph.has_edge(v_fault, u_fault):
                self.current_graph.edges[v_fault, u_fault]['in_service'] = False
            self.steps_since_last_fault = 0
        else:
            self.steps_since_last_fault += 1
        self._last_blackout_status = None
        served_load_count, blackout_status = simulate_grid_state(self.current_graph)
        self._last_blackout_status = blackout_status
        next_state_representation = self._get_state_representation()
        reward = self._calculate_reward(blackout_status, action_cost)
        critical_nodes = [n for n, data in self.current_graph.nodes(data=True)
                           if data.get('critical', False) and data['type'] == 'Load Zone']
        terminated = False
        if critical_nodes:
             terminated = all(blackout_status.get(n, True) for n in critical_nodes)
        truncated = False
        info = {
            'served_load_count': served_load_count,
            'blackout_status': blackout_status,
            'faulted_line': faulted_line_key,
            'healed_lines': healed_lines_info,
            'action_taken': self.action_map[action_index],
            'action_cost': action_cost
        }
        return next_state_representation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_graph = copy.deepcopy(self.nx_graph)
        self.steps_since_last_fault = 0
        self._last_blackout_status = None
        _, initial_blackout_status = simulate_grid_state(self.current_graph)
        self._last_blackout_status = initial_blackout_status
        initial_state_representation = self._get_state_representation()
        return initial_state_representation, {}

    def setup_render(self):
        if not plt.isinteractive():
             plt.ion()
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(16, 10))
            self.fig.canvas.manager.set_window_title("Power Grid Simulation (DQN - Connectivity Only)")
            plt.show(block=False)
            plt.pause(0.1)
        else:
             self.fig.clf()
             self.ax = self.fig.add_subplot(111)

    def render(self, mode='human', step_info=""):
        if mode != 'human' or self.current_graph is None:
            return
        if self.fig is None or self.ax is None or not plt.fignum_exists(self.fig.number):
             self.setup_render()
        self.ax.clear()
        if self._last_blackout_status is None:
             _, blackout_status = simulate_grid_state(self.current_graph)
             self._last_blackout_status = blackout_status
        else:
             blackout_status = self._last_blackout_status
        node_colors = {}
        node_shapes = {}
        node_sizes = {}
        node_lists = defaultdict(list)
        for node, data in self.current_graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            node_lists[node_type].append(node)
            color, shape, size = 'gray', 'd', 1000
            if node_type == 'Generator':
                color, shape, size = 'lightgreen', 's', 2000
            elif node_type == 'Substation':
                color, shape, size = 'lightblue', 'o', 1500
            elif node_type == 'Load Zone':
                shape, size = '^', 1800
                is_blacked_out = blackout_status.get(node, True)
                is_critical = data.get('critical', False)
                if is_blacked_out: color = 'red'
                elif is_critical: color = 'salmon'
                else: color = 'orange'
            node_colors[node], node_shapes[node], node_sizes[node] = color, shape, size
        for node_type, nodes in node_lists.items():
             if not nodes: continue
             type_shapes = [node_shapes.get(n, 'o') for n in nodes]
             type_sizes = [node_sizes.get(n, 1000) for n in nodes]
             type_colors = [node_colors.get(n, 'gray') for n in nodes]
             representative_shape = type_shapes[0]
             nx.draw_networkx_nodes(self.current_graph, self.pos, ax=self.ax, nodelist=nodes,
                                     node_size=type_sizes, node_color=type_colors, node_shape=representative_shape)
        edge_colors = []
        edge_widths = []
        edge_styles = []
        edge_labels_dict = {}
        edges_to_draw = list(self.current_graph.edges(data=True))
        for u, v, data in edges_to_draw:
            label = data.get('label', f'{u[-2:]}-{v[-2:]}')
            in_service = data.get('in_service', False)
            component = data.get('component', 'line')
            style, width, color = 'solid', 1.5, 'gray'
            if component == 'feeder': color, width, style = 'darkorange', 1.0, 'solid'
            elif component == 'transmission': color, width = 'black', 2.5
            elif component == 'tie_line': color, width, style = 'blue', 2.0, 'dashdot'
            else: color, width = 'dimgray', 1.5
            edge_label_text = label
            if not in_service:
                color, style, width = 'red', 'dotted', 0.8
                edge_label_text = f"({label} OFF)"
            edge_colors.append(color)
            edge_widths.append(width)
            edge_styles.append(style)
            edge_labels_dict[(u, v)] = edge_label_text
        node_size_list = [node_sizes.get(n, 1000) for n in self.current_graph.nodes()]
        nx.draw_networkx_edges(self.current_graph, self.pos, ax=self.ax,
                               edgelist=[(u, v) for u, v, _ in edges_to_draw],
                               edge_color=edge_colors,
                               width=edge_widths,
                               style=edge_styles,
                               arrowstyle='-',
                               node_size=node_size_list)
        nx.draw_networkx_labels(self.current_graph, self.pos, ax=self.ax, font_size=8, font_weight='normal')
        nx.draw_networkx_edge_labels(self.current_graph, self.pos, ax=self.ax,
                                     edge_labels=edge_labels_dict,
                                     font_color='black', font_size=6, alpha=0.9,
                                     label_pos=0.5, rotate=False)
        self.ax.set_title(f"Power Grid Simulation (DQN - Connectivity)\n{step_info}", fontsize=10)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        try:
             self.fig.canvas.draw_idle()
             self.fig.canvas.flush_events()
        except Exception as e:
             print(f"Error during canvas draw/flush: {e}")
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            print("Closing plot window.")
            plt.close(self.fig)
            self.fig = None
            self.ax = None

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_critical_blackouts = []
        self._current_reward_sum = 0
        self._current_blackouts_sum = 0
        self._episode_start_step = 0
        self._ep_count = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        terminated = self.locals['terminations'][0]
        truncated = self.locals['truncations'][0]
        info = self.locals['infos'][0]
        self._current_reward_sum += reward
        crit_blackouts_this_step = sum(1 for node, blacked_out in info['blackout_status'].items()
                                       if blacked_out and self.training_env.get_attr('current_graph')[0].nodes[node].get('critical', False))
        self._current_blackouts_sum += crit_blackouts_this_step
        if terminated or truncated:
            episode_length = self.num_timesteps - self._episode_start_step
            avg_blackouts = self._current_blackouts_sum / episode_length if episode_length > 0 else 0
            self.episode_rewards.append(self._current_reward_sum)
            self.episode_critical_blackouts.append(avg_blackouts)
            self._ep_count += 1
            if self._ep_count % 100 == 0:
                 avg_reward_last_100 = np.mean(self.episode_rewards[-100:])
                 avg_blackout_last_100 = np.mean(self.episode_critical_blackouts[-100:])
                 if self.verbose > 0:
                      print(f"Episodes: {self._ep_count}, Timesteps: {self.num_timesteps}, "
                            f"Avg Rew (100): {avg_reward_last_100:.1f}, "
                            f"Avg Crit Blkt (100): {avg_blackout_last_100:.2f}")
            self._current_reward_sum = 0
            self._current_blackouts_sum = 0
            self._episode_start_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
         self.model.save(f"./{MODEL_FILENAME}_final")
         if self.verbose > 0:
              print(f"Saving final model to ./{MODEL_FILENAME}_final.zip")

if __name__ == "__main__":
    TRAIN_FAULT_RATE = 0.12
    AUTO_HEAL_RATE = 0.10
    DEMO_FAULT_RATE = 0.18
    DEMO_AUTO_HEAL_RATE = 0.15
    MAX_STEPS_PER_EPISODE = 70
    TOTAL_TRAINING_TIMESTEPS = 500000

    plt.ion()

    env = make_vec_env(lambda: SimplePowerGridEnv(G, initial_edge_states,
                                                 fault_rate=TRAIN_FAULT_RATE,
                                                 auto_heal_rate=AUTO_HEAL_RATE),
                       n_envs=1)

    model = None
    load_model_path = f"./{MODEL_FILENAME}_final.zip"

    load_model = input(f"Load trained DQN model from {load_model_path}? (y/n): ").strip().lower() == 'y'

    if load_model and os.path.exists(load_model_path):
        try:
            model = DQN.load(load_model_path, env=env)
            print(f"Successfully loaded model from {load_model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Starting training from scratch.")
            model = None
    else:
         if load_model:
              print(f"Model file not found at {load_model_path}. Starting training from scratch.")

    if model is None:
         print("Initializing new DQN model.")
         model = DQN("MlpPolicy", env,
                     learning_rate=0.0005,
                     buffer_size=50000,
                     learning_starts=1000,
                     batch_size=64,
                     gamma=0.99,
                     tau=1.0,
                     train_freq=4,
                     target_update_interval=2000,
                     exploration_fraction=0.15,
                     exploration_final_eps=0.02,
                     seed=42,
                     verbose=1)

    perform_training = input("Perform additional training? (y/n): ").strip().lower() == 'y'

    if perform_training:
        print(f"\n--- Training (DQN - Connectivity) for {TOTAL_TRAINING_TIMESTEPS} timesteps ---")
        plt.ioff()
        callback = CustomCallback(verbose=1)
        training_start_time = time.time()
        try:
             model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS, callback=callback)
        except Exception as e:
             print(f"An error occurred during training: {e}")
             print("Attempting to save current model state before exiting.")
             try:
                  model.save(f"./{MODEL_FILENAME}_interrupted")
                  print(f"Saved interrupted model state to ./{MODEL_FILENAME}_interrupted.zip")
             except Exception as save_e:
                  print(f"Failed to save interrupted model state: {save_e}")
        training_end_time = time.time()
        print(f"\nTraining finished after {training_end_time - training_start_time:.2f} seconds.")
        print(f"Final model saved to ./{MODEL_FILENAME}_final.zip")
    else:
        print("Skipping training.")
        if not load_model and os.path.exists(f"./{MODEL_FILENAME}_final.zip"):
             print("Loading final trained model for demonstration.")
             try:
                  model = DQN.load(f"./{MODEL_FILENAME}_final.zip", env=env)
             except Exception as e:
                  print(f"Error loading final model: {e}. Cannot run demonstration.")
                  model = None

    if model is not None:
        print("\n--- Demonstrating Learned Policy (DQN - Connectivity) ---")
        if isinstance(env, gym.vector.SyncVectorEnv):
             demo_env = env.envs[0]
        else:
             demo_env = env
        demo_env.fault_rate = DEMO_FAULT_RATE
        demo_env.auto_heal_rate = DEMO_AUTO_HEAL_RATE
        print(f"Demonstration Params: Fault Rate={demo_env.fault_rate}, Heal Rate={demo_env.auto_heal_rate}")
        plt.ion()
        print("Opening demonstration window...")
        demo_env.setup_render()
        obs, _ = demo_env.reset()
        initial_served_count, initial_blackout_status = simulate_grid_state(demo_env.current_graph)
        initial_crit_blackouts = sum(1 for node, blacked_out in initial_blackout_status.items()
                                     if blacked_out and demo_env.current_graph.nodes[node].get('critical', False))
        initial_info_str = (f"Step: 0 (Initial State)\n"
                            f"Crit Blackouts: {initial_crit_blackouts}")
        print("Rendering Initial State...")
        demo_env.render(step_info=initial_info_str)
        time.sleep(3)
        cumulative_reward_demo = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            print(f"\n--- Demo Step {step + 1} ---")
            action, _states = model.predict(obs, deterministic=True)
            action_idx = action.item() if isinstance(action, np.ndarray) else action
            action_details = demo_env.action_map[action_idx]
            action_str = f"Action {action_idx}: {action_details[0]}"
            target_info = ""
            if action_details[1] is not None:
                u, v = action_details[1]
                edge_data = demo_env.current_graph.get_edge_data(u, v, default={})
                line_label = edge_data.get('label', f"{u}-{v} (edge data missing)")
                target_info = f"Target: {line_label}"
            print(f"{action_str} {target_info}")
            obs, reward, terminated, truncated, info = demo_env.step(action_idx)
            cumulative_reward_demo += reward
            crit_blackouts_this_step = sum(1 for node, blacked_out in info['blackout_status'].items()
                                            if blacked_out and demo_env.current_graph.nodes[node].get('critical', False))
            info_str = (f"Step: {step + 1}, Action: {action_details[0]}, Reward: {reward:.1f} (Total: {cumulative_reward_demo:.1f})\n"
                        f"Crit Blackouts: {crit_blackouts_this_step}")
            event_str = ""
            faulted_line_key = info.get('faulted_line')
            healed_lines_info = info.get('healed_lines')
            if faulted_line_key:
                 u_f, v_f = faulted_line_key
                 edge_data_f = demo_env.current_graph.get_edge_data(u_f, v_f, default={})
                 fault_label = edge_data_f.get('label', str(faulted_line_key))
                 event_str += f"\nFAULT: {fault_label} tripped!"
                 print(f"FAULT: {fault_label} tripped!")
            if healed_lines_info:
                healed_labels = []
                processed_pairs = set()
                for u_h, v_h in healed_lines_info:
                    pair = tuple(sorted((u_h, v_h)))
                    if pair not in processed_pairs:
                         edge_data_h = demo_env.current_graph.get_edge_data(u_h, v_h, default={})
                         label_h = edge_data_h.get('label', f"{u_h}-{v_h}")
                         healed_labels.append(label_h)
                         processed_pairs.add(pair)
                if healed_labels:
                     heal_info = f"HEALED: {', '.join(map(str, healed_labels))}"
                     event_str += f"\n{heal_info}"
                     print(heal_info)
            info_str += event_str
            demo_env.render(step_info=info_str)
            time.sleep(1.0)
            if terminated or truncated:
                reason = "All critical loads out" if terminated else "Max steps reached"
                print(f"Episode finished at step {step + 1}. Reason: {reason}")
                final_info_str = info_str + f"\nEpisode {'Terminated!' if terminated else 'Truncated!'}"
                demo_env.render(step_info=final_info_str)
                time.sleep(5)
                break
        print("\nDemonstration finished.")
        demo_env.close()
        print("Plot window closed.")

    plt.ioff()