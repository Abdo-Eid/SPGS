"""
Main Script for Training and Demonstrating the SARSA Agent on the Power Grid Environment.

This script provides a command-line interface to:
1. Train the SARSA agent (`--mode train`): Runs the agent through multiple episodes,
   updates its Q-table, and periodically saves the table.
2. Demonstrate a trained agent (`--mode demo`): Loads a saved Q-table and runs the
   agent greedily in the environment, rendering the steps based on chosen mode.

Command-line arguments control the mode, training parameters (episodes, learning rate,
discount factor, epsilon), demonstration parameters (steps, render mode), and
file paths for saving/loading the Q-table.
"""

import argparse
import numpy as np
from tqdm import tqdm # Progress bar visualization
import time
import sys
import traceback # For printing detailed error information

# Import configuration constants (e.g., N_GENS for rendering)
try:
    from conf import N_EM_GENS, N_MAIN_GENS
except ImportError:
    print("Error: Could not import constants from conf.py. Ensure it exists.")
    # Provide default values to allow script to potentially continue or fail later
    N_MAIN_GENS = 3
    N_EM_GENS = 3
    print("Warning: Using default N_MAIN_GENS=3, N_EM_GENS=3.")


# --- Ensure local imports work ---
# Attempt to import the custom environment and agent classes.
# Provide helpful error messages if imports fail.
try:
    from grid_env import PowerGridEnv
    from sarsa_agent import SARSAgent # Import the SARSA agent
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure grid_env.py, sarsa_agent.py, utilites.py, and conf.py")
    print("are in the same directory as main.py or accessible in the Python path.")
    sys.exit(1)
# --- End Local Import Handling ---

def run_training(env, agent, episodes, save_file):
    """
    Executes the SARSA training loop for a specified number of episodes.

    Args:
        env (PowerGridEnv): The environment instance.
        agent (SARSAgent): The SARSA agent instance.
        episodes (int): The total number of training episodes to run.
        save_file (str): The filename where the Q-table will be saved periodically and finally.
    """
    print(f"Starting SARSA training for {episodes} episodes...")
    total_rewards = [] # List to store total reward per episode
    # Configure intervals for reporting progress and saving the Q-table
    report_interval = max(1, episodes // 50) # Report progress roughly 50 times
    save_interval = max(10, episodes // 10) # Save Q-table roughly 10 times + final save

    # --- Training Loop ---
    for episode in tqdm(range(episodes), desc="Training Progress", unit="episode"):
        # Reset environment for a new episode
        try:
            obs, _ = env.reset()
        except Exception as e:
            tqdm.write(f"\nERROR: env.reset() failed at start of episode {episode+1}. Error: {e}")
            traceback.print_exc()
            break # Stop training if reset fails critically

        # Ensure initial observation is valid before discretizing
        if obs is None:
             tqdm.write(f"\nERROR: env.reset() returned None observation at episode {episode+1}. Aborting training.")
             break

        # Discretize the initial observation to get the starting state
        try:
            state = agent.discretize_state(obs)
        except Exception as e:
            tqdm.write(f"\nERROR: Failed to discretize initial state at episode {episode+1}. Obs: {obs}. Error: {e}")
            traceback.print_exc()
            break # Stop training if discretization fails critically

        # Choose the first action 'a' using the agent's policy (epsilon-greedy)
        try:
            action_tuple = agent.choose_action(state)
        except Exception as e:
            tqdm.write(f"\nERROR: Failed to choose initial action at episode {episode+1}. State: {state}. Error: {e}")
            traceback.print_exc()
            break

        # Initialize episode variables
        done = False # Flag indicating episode termination
        episode_reward = 0.0
        step_count = 0

        # --- Inner Loop (Steps within an episode) ---
        while not done:
            step_count += 1
            # --- Execute Action and Observe Outcome ---
            # Take action 'a', observe reward 'r' and next observation 'next_obs'
            try:
                next_obs, reward, terminated, truncated, info = env.step(action_tuple)
                done = terminated or truncated # Combine termination flags
            except Exception as e:
                 # Handle errors during environment step (e.g., internal env error)
                 tqdm.write(f"\nERROR: env.step() failed at episode {episode+1}, step {step_count}. "
                            f"Action: {action_tuple}. Error: {e}")
                 traceback.print_exc()
                 # Decide how to handle: end episode, penalize, try to recover?
                 done = True # End this episode prematurely
                 reward = -1000 # Assign a large penalty? (Optional)
                 next_obs = obs # Use previous observation to prevent crash in discretization
                 terminated = True # Ensure loop exit

            # --- Handle Potentially Invalid Next Observation ---
            if next_obs is None:
                 tqdm.write(f"\nERROR: env.step() returned None observation at episode {episode+1}, step {step_count}. "
                            f"Ending episode.")
                 done = True
                 next_obs = obs # Use previous observation to avoid crash

            # --- Discretize Next State ---
            # Discretize the observed 'next_obs' to get the next state 'next_state'
            try:
                 next_state = agent.discretize_state(next_obs)
            except Exception as e:
                tqdm.write(f"\nERROR: Failed to discretize next state at episode {episode+1}, step {step_count}. "
                           f"Next Obs: {next_obs}. Error: {e}")
                traceback.print_exc()
                done = True # End episode
                next_state = state # Use previous state to avoid crash

            # --- Choose Next Action ---
            # Choose the next action 'next_action_tuple' (a') from 'next_state' using the policy.
            # This is needed for the SARSA update, even if the episode just ended.
            try:
                 next_action_tuple = agent.choose_action(next_state)
            except Exception as e:
                 tqdm.write(f"\nERROR: Failed to choose next action at episode {episode+1}, step {step_count}. "
                            f"Next State: {next_state}. Error: {e}")
                 traceback.print_exc()
                 # If choosing next action fails, we might not be able to update.
                 # Option: Skip update or use a default next_action? Let's skip update for safety.
                 done = True # End episode if we can't choose next action


            # --- SARSA Update ---
            # Update the Q-table using the experience tuple (s, a, r, s', a')
            # Only perform update if the step didn't fail before getting valid next state/action
            if not (done and next_state == state and step_count > 1): # Avoid redundant update if error prevented progress
                 try:
                      agent.update(state, action_tuple, reward, next_state, next_action_tuple)
                 except Exception as e:
                      tqdm.write(f"\nERROR: agent.update() failed at episode {episode+1}, step {step_count}. Error: {e}")
                      traceback.print_exc()
                      # Decide whether to stop training or just log the error

            # --- Prepare for Next Iteration ---
            # Update state and action for the next loop iteration
            state = next_state
            action_tuple = next_action_tuple
            episode_reward += reward

            # Max steps per episode check (optional, Gymnasium handles via truncated)
            # if step_count >= MAX_STEPS_PER_EPISODE:
            #     done = True
            #     truncated = True # Indicate truncation due to step limit

        # --- End of Episode ---
        total_rewards.append(episode_reward)

        # --- Reporting and Saving ---
        # Print average reward periodically
        if (episode + 1) % report_interval == 0:
            avg_reward = np.mean(total_rewards[-report_interval:])
            # Use tqdm.write to avoid interfering with the progress bar
            tqdm.write(f"Episode {episode+1}/{episodes} | Avg Reward (last {report_interval}): {avg_reward:.2f}")

        # Save Q-table periodically
        if (episode + 1) % save_interval == 0:
             tqdm.write(f"Saving Q-table at episode {episode+1}...")
             agent.save_q_table(save_file)

    # --- End of Training ---
    print("\nTraining finished.")
    print("Saving final Q-table...")
    agent.save_q_table(save_file) # Final save of the Q-table


def run_demonstration(env, agent, demo_steps, render_mode):
    """
    Runs the trained agent in the environment for demonstration purposes.

    The agent acts greedily (epsilon = 0) based on the loaded Q-table.
    Renders the environment state at each step according to the specified `render_mode`.

    Args:
        env (PowerGridEnv): The environment instance, potentially configured for rendering.
        agent (SARSAgent): The trained SARSA agent with a loaded Q-table.
        demo_steps (int): The maximum number of steps to run the demonstration for.
        render_mode (str): The rendering mode ('human', 'terminal', or 'none').
    """
    print(f"\nStarting demonstration for up to {demo_steps} steps...")
    print(f"Render mode: {render_mode}")
    agent.epsilon = 0.0 # Set epsilon to 0 for greedy actions (exploitation only)

    try:
        # Reset environment for demonstration
        obs, _ = env.reset()
        total_reward = 0.0
        steps_taken = 0

        # Check initial observation
        if obs is None:
             print("ERROR: env.reset() returned None observation at start of demonstration.")
             return

        # --- Demonstration Loop ---
        for _ in range(demo_steps):
            steps_taken += 1 # Increment step counter first

            # Discretize the current observation
            try:
                state = agent.discretize_state(obs)
            except Exception as e:
                print(f"\nERROR: Failed to discretize state at demo step {steps_taken}. Obs: {obs}. Error: {e}")
                traceback.print_exc()
                break # Stop demonstration on error

            # Choose the best action greedily (epsilon = 0)
            try:
                action_tuple = agent.choose_action(state)
            except Exception as e:
                print(f"\nERROR: Failed to choose action at demo step {steps_taken}. State: {state}. Error: {e}")
                traceback.print_exc()
                break

            # Execute the action in the environment
            try:
                obs, reward, terminated, truncated, info = env.step(action_tuple)
                # Note: Rendering for 'human' mode is now handled *within* env.step()
                # if env.render_mode was set to 'human' during initialization.
            except Exception as e:
                 print(f"\nERROR: env.step() failed at demo step {steps_taken}. Action: {action_tuple}. Error: {e}")
                 traceback.print_exc()
                 break # Stop demonstration on error

            total_reward += reward

            # --- Terminal Rendering Logic (if requested) ---
            # This provides step-by-step text output without clearing the screen.
            if render_mode == 'terminal':
                print(f"\n--- Step {steps_taken} ---")
                if not info: # Check if info dictionary is available
                    print("  Info dictionary not available for this step.")
                else:
                    # Safely format action and reward
                    action_str = str(action_tuple) if action_tuple is not None else "N/A"
                    reward_str = f"{reward:.2f}" if reward is not None else "N/A"
                    print(f"Action Taken: {action_str}")
                    print(f"Step Reward: {reward_str}")

                    # Safely get and format key info values
                    current_time = info.get('current_time', 'N/A')
                    deficit = info.get('power_balance_deficit_MW', 'N/A')
                    batt_soc = info.get('battery_soc_MWh', 'N/A')
                    batt_mode_map = {0: "Idle", 1: "Discharge", 2: "Charge"}
                    batt_mode = batt_mode_map.get(info.get('battery_action_mode', -1), 'N/A')
                    main_online = info.get('main_gen_online', [])
                    em_online = info.get('emergency_gens_online', [])

                    print(f"Time: {current_time:.1f} hr" if isinstance(current_time, (int, float)) else f"Time: {current_time}")
                    print(f"Deficit: {deficit:.1f} MW" if isinstance(deficit, (int, float)) else f"Deficit: {deficit}")
                    print(f"Battery SoC: {batt_soc:.1f} MWh ({batt_mode})" if isinstance(batt_soc, (int, float)) else f"Battery SoC: {batt_soc} ({batt_mode})")
                    print(f"Main Gens Online: {sum(main_online)}/{N_MAIN_GENS}")
                    print(f"EM Gens Online: {sum(em_online)}/{N_EM_GENS}")
                    if info.get('critical_failure', False):
                        print("  \033[91m[CRITICAL FAILURE] High priority load not met!\033[0m") # Red text

            # --- Check for None observation after step ---
            if obs is None:
                 print(f"\nERROR: env.step() returned None observation at demo step {steps_taken}. Stopping demonstration.")
                 break

            # --- Check for Episode End ---
            if terminated or truncated:
                print(f"\nEpisode finished within demonstration period (at step {steps_taken}).")
                if steps_taken < demo_steps:
                    print("Resetting environment for potential continuation (if needed)...")
                    # Reset if the demo is supposed to continue beyond one episode,
                    # although typically demo runs for a fixed number of steps total.
                    obs, _ = env.reset()
                    if obs is None:
                        print("\nERROR: env.reset() returned None observation after episode finish during demo.")
                        break
                else:
                    break # Exit loop if requested steps reached

            # Optional delay for terminal mode if needed (human mode handles its own delay)
            # if render_mode == 'terminal': time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    finally:
        # --- End of Demonstration ---
        print(f"\nDemonstration finished after {steps_taken} steps.")
        print(f"Total reward accumulated during demo: {total_reward:.2f}")


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Argument Parsing ---
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Train or run SARSA agent for PowerGridEnv.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'demo'],
                        help="Execution mode: 'train' for training, 'demo' for demonstration.")
    # Training arguments
    parser.add_argument('--episodes', type=int, default=5000,
                        help="Number of episodes for training (default: 5000).")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate (alpha) for SARSA update (default: 0.1).")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor (gamma) for future rewards (default: 0.99).")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="Constant epsilon value for epsilon-greedy exploration during training (default: 0.1).")
    # Demonstration arguments
    parser.add_argument('--demosteps', type=int, default=50,
                        help="Maximum number of steps for demonstration mode (default: 50).")
    parser.add_argument('--render_mode', type=str, choices=['human', 'terminal', 'none'], default='human',
                        help="Rendering mode for demonstration: 'human' (dashboard), 'terminal' (step summary), 'none' (no per-step output). Default: 'human'.")
    parser.add_argument('--render_time', type=float, default=2.5,
                        help="Time in seconds to pause between steps when render_mode is 'human' (default: 2.5).")
    # Common arguments
    parser.add_argument('--load', action='store_true',
                        help="Load a pre-trained Q-table from --save_file before starting training or demonstration.")
    parser.add_argument('--save_file', type=str, default="sarsa_q_table.pkl",
                        help="Filename for saving (during training) and loading the Q-table (default: sarsa_q_table.pkl).")

    args = parser.parse_args() # Parse the command-line arguments

    # --- Environment Setup ---
    # Initialize the PowerGridEnv.
    # Crucially, set the render_mode for the environment *only* if needed ('human' mode).
    # For training or 'terminal'/'none' demo, the environment itself doesn't need
    # the render_mode set, as rendering is handled externally or not at all.
    render_mode_for_env = 'human' if args.mode == 'demo' and args.render_mode == 'human' else None
    env = None # Initialize to None for finally block safety
    try:
        print(f"Initializing PowerGridEnv (render_mode='{render_mode_for_env}')...")
        env = PowerGridEnv(render_mode=render_mode_for_env,RENDER_SLEEP_TIME=args.render_time)
        print("Environment initialized.")
    except Exception as e:
        print(f"FATAL: Error initializing environment: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Agent Setup ---
    # Initialize the SARSA agent, passing the environment and hyperparameters.
    # Note: Discretization bins are crucial and likely need tuning in sarsa_agent.py!
    agent = None # Initialize to None for finally block safety
    try:
        print("Initializing SARSA agent...")
        agent = SARSAgent(env,
                          learning_rate=args.lr,
                          discount_factor=args.gamma,
                          epsilon=args.epsilon) # Epsilon is constant during training here
        print("Agent initialized.")
    except Exception as e:
        print(f"FATAL: Error initializing SARSA agent: {e}")
        traceback.print_exc()
        if env: env.close() # Close env if it was opened
        sys.exit(1)

    # --- Load Q-table if requested ---
    q_table_loaded = False
    if args.load:
        print(f"Attempting to load Q-table from: {args.save_file}")
        q_table_loaded = agent.load_q_table(args.save_file)
        if not q_table_loaded:
            # If loading failed, handle differently based on mode
            if args.mode == 'demo':
                 print(f"ERROR: Failed to load Q-table '{args.save_file}', which is required for demo mode.")
                 env.close()
                 sys.exit(1)
            elif args.mode == 'train':
                 print(f"WARN: Failed to load Q-table '{args.save_file}'. Starting training from scratch.")
        else:
             print("Q-table loaded successfully.")

    # --- Check prerequisites for demo mode ---
    if args.mode == 'demo':
        # Demo mode requires a Q-table to be loaded successfully.
        if not args.load: # Check if --load flag was explicitly used
             print("ERROR: Demo mode requires loading a Q-table. Use the --load flag.")
             env.close()
             sys.exit(1)
        if not q_table_loaded: # Check if loading actually succeeded
             print("ERROR: Cannot run demo because Q-table failed to load (see previous errors).")
             env.close()
             sys.exit(1)
        # If checks pass, proceed to demonstration.

    # --- Run Selected Mode ---
    try:
        if args.mode == 'train':
            # Start the training process
            run_training(env, agent, args.episodes, args.save_file)
        elif args.mode == 'demo':
            # Start the demonstration process
            # Prerequisites (load flag used, load successful) already checked
            run_demonstration(env, agent, args.demosteps, args.render_mode)

    except Exception as e:
        # Catch any unexpected errors during training or demonstration runs
        print(f"\nFATAL: An unexpected error occurred during {args.mode} execution:")
        print(e)
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # --- Cleanup ---
        # Ensure the environment is properly closed regardless of errors
        if env:
            print("Closing environment...")
            env.close()
        print("Execution finished.")
