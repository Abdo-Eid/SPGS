import matplotlib.pyplot as plt
import numpy as np
import time
import random
from collections import defaultdict
import networkx as nx # Needed for plotting

# Import components from other modules
from grid_topology import create_grid_topology
from grid_utils import save_q_table, load_q_table, Q_TABLE_FILENAME
from grid_simulation import simulate_grid_state # Needed for initial state render
from rl_environment import SimplePowerGridEnv

# --- Plotting and Visualization Functions ---

# Global figure and axes handles for live plotting
fig = None
ax = None

def setup_visualization():
    """Initializes the Matplotlib figure and axes for live rendering."""
    global fig, ax
    # Turn on interactive mode BEFORE creating the plot
    if not plt.isinteractive():
         plt.ion()

    if fig is None or not plt.fignum_exists(fig.number):
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111)
        fig.canvas.manager.set_window_title("Power Grid Simulation")
        # Show the figure window immediately but don't block execution
        plt.show(block=False)
        plt.pause(0.1) # Allow time for window to appear
    else:
         # If figure exists, clear it completely for the new render setup
         fig.clf()
         ax = fig.add_subplot(111)
    print("Visualization window initialized.")


def render_grid_state(env, pos, step_info=""):
    """
    Renders the current grid state using Matplotlib.

    Args:
        env (SimplePowerGridEnv): The environment instance containing the current state.
        pos (dict): Dictionary mapping node names to (x, y) positions.
        step_info (str): String containing information about the current step/state.
    """
    global fig, ax

    if env.current_graph is None:
        print("Cannot render, environment graph not initialized.")
        return

    # Ensure the plot window is set up
    if fig is None or ax is None or not plt.fignum_exists(fig.number):
         print("Setting up visualization window for render...")
         setup_visualization() # Create fig/ax if needed

    # --- Force Redraw Strategy ---
    # Clear the entire figure before drawing
    fig.clf()
    # Add new axes after clearing
    ax = fig.add_subplot(111)

    # Get current simulation state for coloring/styling
    # Use cached status from env if available, otherwise simulate
    if env._last_blackout_status is None:
         # Simulate if cache is empty (e.g., first render after reset)
         served_load_mw, blackout_status, overloaded_lines = simulate_grid_state(env.current_graph)
         # Cache the results back into the environment object if needed elsewhere,
         # though it's primarily used for rendering here.
         # env._last_blackout_status = blackout_status # Optional caching
    else:
         # Use cached status from the last env.step() call
         blackout_status = env._last_blackout_status
         # Still need to recalculate overloads as they depend on current topology/load
         # Note: This recalculates simulation logic slightly redundantly if called after step()
         # where info dict already has this. Could pass info dict instead.
         _, _, overloaded_lines = simulate_grid_state(env.current_graph)

    # --- Node Drawing ---
    node_colors = {}
    node_shapes = {}
    node_sizes = {}
    node_lists = defaultdict(list) # Group nodes by type for drawing

    for node, data in env.current_graph.nodes(data=True):
        node_type = data.get('type', 'Unknown') # Safely get type
        node_lists[node_type].append(node)

        # Assign visual properties based on node type and status
        if node_type == 'Generator':
            node_colors[node], node_shapes[node], node_sizes[node] = 'lightgreen', 's', 3000 # Square
        elif node_type == 'Substation':
            node_colors[node], node_shapes[node], node_sizes[node] = 'lightblue', 'o', 2500 # Circle
        elif node_type == 'Load Zone':
            node_shapes[node], node_sizes[node] = '^', 2000 # Triangle Up
            is_blacked_out = blackout_status.get(node, True) # Default to True if not found
            is_critical = data.get('critical', False)
            if is_blacked_out:
                node_colors[node] = 'red'
            elif is_critical:
                node_colors[node] = 'salmon' # Critical but served
            else:
                node_colors[node] = 'orange' # Non-critical and served
        else: # Default for unknown types
            node_colors[node], node_shapes[node], node_sizes[node] = 'gray', 'd', 1000 # Diamond

    # Draw nodes type by type to handle different shapes correctly
    for node_type, nodes in node_lists.items():
         if not nodes: continue # Skip if no nodes of this type
         # Ensure all nodes in the list exist in the position mapping
         valid_nodes = [n for n in nodes if n in pos]
         if not valid_nodes: continue # Skip if no valid nodes with positions

         type_shape = node_shapes[valid_nodes[0]] # Get shape from first valid node
         nx.draw_networkx_nodes(env.current_graph, pos, ax=ax, nodelist=valid_nodes,
                                 node_size=[node_sizes[n] for n in valid_nodes],
                                 node_color=[node_colors[n] for n in valid_nodes],
                                 node_shape=type_shape)

    # --- Edge Drawing ---
    edge_colors = []
    edge_widths = []
    edge_labels_dict = {}
    edges_to_draw = list(env.current_graph.edges(data=True)) # Get edges with data

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
            # edge_labels_dict[(u, v)] = label # Optional: Show label only for off/overloaded
            edge_labels_dict[(u, v)] = "" # Keep plot cleaner

    # Draw the edges using the collected styles
    # Ensure the edgelist used matches the order of colors/widths
    nx.draw_networkx_edges(env.current_graph, pos, ax=ax,
                           edgelist=[(u,v) for u,v,_ in edges_to_draw], # Pass just (u,v) tuples
                           edge_color=edge_colors,
                           width=edge_widths,
                           arrowstyle='->', arrowsize=15)

    # --- Label Drawing ---
    # Draw node labels (names)
    nx.draw_networkx_labels(env.current_graph, pos, ax=ax, font_size=9, font_weight='bold')

    # Draw edge labels (status/overload info) - positioned slightly above center
    nx.draw_networkx_edge_labels(env.current_graph, pos, ax=ax,
                                 edge_labels=edge_labels_dict,
                                 font_color='darkred', font_size=7, alpha=0.9,
                                 label_pos=0.6) # Adjust label position slightly


    # --- Final Touches ---
    ax.set_title(f"Power Grid Simulation\n{step_info}", fontsize=12)
    ax.set_axis_off() # Hide axes
    fig.tight_layout() # Adjust layout to prevent overlap

    # --- Crucial Update Steps for Interactive Plot ---
    try:
         fig.canvas.draw_idle() # Request redraw efficiently
         fig.canvas.flush_events() # Process GUI events to make plot responsive
    except Exception as e:
         print(f"Error during canvas draw/flush: {e}") # Catch potential GUI errors

    plt.pause(0.01) # VERY IMPORTANT: Small pause allows plot to update visually

def close_visualization():
    """Closes the Matplotlib plot window."""
    global fig, ax
    if fig is not None:
        print("Closing visualization window.")
        plt.close(fig)
        fig = None
        ax = None
    # Turn off interactive mode when done
    # plt.ioff() # Comment this out if other plots might be generated later

def plot_training_results(num_episodes, rewards, loads, overloads, blackouts):
    """Generates static plots of training metrics."""
    episodes = range(num_episodes)
    plt.ioff() # Ensure non-interactive mode for saving/showing static plots

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode during Training")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, loads)
    plt.xlabel("Episode")
    plt.ylabel("Average Served Load (MW)")
    plt.title("Average Served Load per Episode during Training")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, overloads)
    plt.xlabel("Episode")
    plt.ylabel("Average Overloaded Lines")
    plt.title("Average Overloaded Lines per Episode during Training")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, blackouts)
    plt.xlabel("Episode")
    plt.ylabel("Avg Critical Blackouts")
    plt.title("Average Critical Blacked Out Load Zones per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    TRAIN_EPISODES = 1500
    MAX_STEPS_PER_EPISODE = 60
    TRAIN_FAULT_RATE = 0.15       # Higher fault rate during training
    TRAIN_AUTO_HEAL_RATE = 0.05
    DEMO_FAULT_RATE = 0.25        # Different rates for demonstration
    DEMO_AUTO_HEAL_RATE = 0.10
    DEMO_PAUSE_TIME = 1.5         # Pause between demo steps (seconds)
    INITIAL_STATE_PAUSE = 3.0     # Pause to view initial state in demo
    FINAL_STATE_PAUSE = 5.0       # Pause after demo finishes

    # --- Setup Grid and Environment ---
    print("1. Creating Grid Topology...")
    G, initial_edge_states, initial_load_demands, node_positions = create_grid_topology()

    print("2. Initializing RL Environment...")
    # Use training parameters first
    env = SimplePowerGridEnv(G, initial_edge_states, initial_load_demands,
                             fault_rate=TRAIN_FAULT_RATE, auto_heal_rate=TRAIN_AUTO_HEAL_RATE)
    action_space_size = env.action_space.n
    state_length = env.observation_space.shape[0]

    # --- Load or Initialize Q-table ---
    print(f"3. Loading/Initializing Q-Table ({Q_TABLE_FILENAME})...")
    load_model = input(f"   Load trained model from {Q_TABLE_FILENAME}? (y/n): ").strip().lower() == 'y'
    loaded_q_table, loaded_state_map, loaded_next_idx = load_q_table(Q_TABLE_FILENAME, action_space_size, state_length)
    # Initialize the environment's Q-table (either loaded or new)
    env.initialize_q_table(loaded_q_table, loaded_state_map, loaded_next_idx)

    # --- Training Phase ---
    perform_training = False
    if load_model and env.next_state_index > 0:
        env.epsilon = 0.1 # Lower epsilon for continued training/demo if loaded
        print(f"   Using loaded Q-table. Epsilon set to {env.epsilon:.2f}.")
        perform_training = input("   Perform additional training? (y/n): ").strip().lower() == 'y'
    else:
        env.epsilon = 1.0 # High epsilon for exploration if starting fresh
        print("   Starting training from scratch.")
        perform_training = True

    if perform_training:
        print("\n--- Starting Training ---")
        # Ensure interactive mode is OFF for potentially faster training without live plots
        plt.ioff()

        episode_rewards = np.zeros(TRAIN_EPISODES)
        served_load_history = np.zeros(TRAIN_EPISODES)
        overload_count_history = np.zeros(TRAIN_EPISODES)
        blackout_count_history = np.zeros(TRAIN_EPISODES) # Track avg critical blackouts

        training_start_time = time.time()
        for episode in range(TRAIN_EPISODES):
            state_tuple, info = env.reset() # Reset env and get initial state
            total_reward = 0
            step_count = 0
            # Track metrics within the episode
            metrics = {'served_load': [], 'overloads': [], 'blackouts': []}

            for step in range(MAX_STEPS_PER_EPISODE):
                action = env.choose_action(state_tuple) # Epsilon-greedy action selection
                next_state_tuple, reward, terminated, truncated, info = env.step(action) # Execute step
                total_reward += reward
                state_tuple = next_state_tuple # Move to next state
                step_count += 1

                # Store step metrics for averaging later
                metrics['served_load'].append(info['served_load_mw'])
                metrics['overloads'].append(len(info['overloaded_lines']))
                # Calculate critical blackouts for this step based on info
                crit_blackouts = sum(1 for node, blacked_out in info['blackout_status'].items()
                                     if blacked_out and env.current_graph.nodes[node].get('critical', False))
                metrics['blackouts'].append(crit_blackouts)

                # Check termination conditions
                if terminated or truncated:
                    break

            # Store episode averages/totals
            episode_rewards[episode] = total_reward
            served_load_history[episode] = np.mean(metrics['served_load']) if metrics['served_load'] else 0
            overload_count_history[episode] = np.mean(metrics['overloads']) if metrics['overloads'] else 0
            blackout_count_history[episode] = np.mean(metrics['blackouts']) if metrics['blackouts'] else 0

            # Print progress periodically
            if (episode + 1) % 100 == 0 or episode == TRAIN_EPISODES - 1:
                end_time = time.time()
                avg_reward_100 = np.mean(episode_rewards[max(0, episode-99):episode+1])
                avg_load_100 = np.mean(served_load_history[max(0, episode-99):episode+1])
                avg_overload_100 = np.mean(overload_count_history[max(0, episode-99):episode+1])
                avg_blackout_100 = np.mean(blackout_count_history[max(0, episode-99):episode+1])
                time_per_100 = end_time - training_start_time
                print(f"Ep {episode+1}/{TRAIN_EPISODES}, Avg Rew (100): {avg_reward_100:.1f}, "
                      f"Avg Load: {avg_load_100:.1f}, Avg Ovrld: {avg_overload_100:.2f}, Avg Crit Blkt: {avg_blackout_100:.2f}, "
                      f"Eps: {env.epsilon:.3f}, States: {env.next_state_index}, Time: {time_per_100:.2f}s")
                training_start_time = time.time() # Reset timer for next block

        print("\nTraining finished.")

        # --- Plot training results ---
        plot_training_results(TRAIN_EPISODES, episode_rewards, served_load_history, overload_count_history, blackout_count_history)

        # --- Save Model ---
        save_model_prompt = input(f"Save trained model to {Q_TABLE_FILENAME}? (y/n): ").strip().lower() == 'y'
        if save_model_prompt:
            save_q_table(env.q_table, env.state_to_index, env.next_state_index, Q_TABLE_FILENAME)
    else:
        print("Skipping training.")
        if not (load_model and env.next_state_index > 0):
             print("Warning: No model loaded and training skipped. Demonstration will use random actions.")
             env.epsilon = 1.0 # Ensure random actions if no Q-table is available
             env.alpha = 0.0 # Ensure no learning happens


    # --- Demonstration Phase ---
    print("\n--- Demonstrating Learned Policy ---")
    # Configure environment for demonstration (no exploration, no learning, potentially different dynamics)
    env.epsilon = 0.0 # Purely greedy policy
    env.alpha = 0.0   # Disable learning updates
    env.fault_rate = DEMO_FAULT_RATE
    env.auto_heal_rate = DEMO_AUTO_HEAL_RATE
    print(f"Demonstration settings: Fault Rate={env.fault_rate}, Heal Rate={env.auto_heal_rate}, Epsilon={env.epsilon}")

    # Set up the visualization window
    setup_visualization()

    # Reset environment for the demo run
    state_tuple, reset_info = env.reset()

    # Render the initial state
    print("Rendering initial state...")
    initial_served_load, initial_blackout_status, initial_overloaded_lines = simulate_grid_state(env.current_graph)
    initial_crit_blackouts = sum(1 for node, blacked_out in initial_blackout_status.items()
                                 if blacked_out and env.current_graph.nodes[node].get('critical', False))
    initial_info_str = (f"Step: 0, Reward: N/A (Initial State)\n"
                        f"Served Load: {initial_served_load:.2f} MW, Overloads: {len(initial_overloaded_lines)}, "
                        f"Crit Blackouts: {initial_crit_blackouts}")
    render_grid_state(env, node_positions, step_info=initial_info_str)
    time.sleep(INITIAL_STATE_PAUSE) # Pause to view initial state

    # Run the demonstration loop
    for step in range(MAX_STEPS_PER_EPISODE):
        print(f"\n--- Demo Step {step+1} ---")

        # Choose action based on the learned policy (epsilon = 0)
        action = env.choose_action(state_tuple)
        action_details = env.action_map[action]

        # Print action details for clarity
        action_str = f"Action {action}: {action_details[0]}"
        target_info = ""
        if action_details[1] is not None:
            if isinstance(action_details[1], tuple): # Line action
                u, v = action_details[1]
                # Safely get edge data and label, provide default if edge missing (shouldn't happen)
                edge_data = env.current_graph.get_edge_data(u, v, {})
                line_label = edge_data.get('label', f"Missing Edge ({u}-{v})")
                target_info = f"Target: {line_label}"
            else: # Load shedding action
                load_node = action_details[1]
                target_info = f"Target: {load_node} (Shed {env.shed_percentage*100:.0f}%)"
        print(f"{action_str} {target_info}")

        # Execute the step in the environment
        next_state_tuple, reward, terminated, truncated, info = env.step(action)

        # Prepare info string for rendering title
        crit_blackouts = sum(1 for node, blacked_out in info['blackout_status'].items()
                             if blacked_out and env.current_graph.nodes[node].get('critical', False))
        info_str = (f"Step: {step+1}, Action: {action_details[0]}, Reward: {reward:.2f}\n"
                    f"Served Load: {info['served_load_mw']:.2f} MW, Overloads: {len(info['overloaded_lines'])}, "
                    f"Crit Blackouts: {crit_blackouts}")

        # Add fault/heal information to the display string and console output
        faulted_line_key = info.get('faulted_line')
        healed_lines_info = info.get('healed_lines')
        if faulted_line_key:
             u,v = faulted_line_key
             # Get data from current state (it's off now, but label should be there)
             edge_data = env.current_graph.get_edge_data(u, v, {})
             fault_label = edge_data.get('label', f"Edge ({u}-{v})") # Use tuple if label missing
             fault_str = f"\nFAULT: {fault_label} tripped!"
             info_str += fault_str
             print(fault_str) # Print to console as well
        if healed_lines_info:
            healed_labels = []
            processed_pairs = set() # Avoid double-reporting bi-directional lines
            for u,v in healed_lines_info:
                pair = tuple(sorted((u, v))) # Treat (u,v) and (v,u) as the same line for reporting
                if pair not in processed_pairs:
                     # Get edge data (it should be 'in_service': True now)
                     edge_data = env.current_graph.get_edge_data(u, v, {})
                     label = edge_data.get('label', f"Edge ({u}-{v})")
                     healed_labels.append(str(label)) # Ensure label is string
                     processed_pairs.add(pair)
            if healed_labels:
                 heal_str = f"\nHEALED: {', '.join(healed_labels)}"
                 info_str += heal_str
                 print(heal_str) # Print to console as well

        # Render the current state with updated info
        render_grid_state(env, node_positions, step_info=info_str)
        time.sleep(DEMO_PAUSE_TIME) # Pause between steps

        state_tuple = next_state_tuple # Update state for next iteration

        # Check if episode ended
        if terminated or truncated:
            end_reason = 'All critical loads out' if terminated else 'Max steps reached'
            print(f"\nEpisode finished at step {step+1}. Reason: {end_reason}")
            final_info_str = info_str + f"\nEpisode {'Terminated!' if terminated else 'Truncated!'}"
            render_grid_state(env, node_positions, step_info=final_info_str) # Render final state
            time.sleep(FINAL_STATE_PAUSE) # Longer pause at the end
            break

    print("\nDemonstration finished.")

    # --- Cleanup ---
    close_visualization() # Close the plot window
    print("Script complete.")
