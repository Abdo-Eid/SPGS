import argparse
import numpy as np
from tqdm import tqdm # Progress bar
import time
import sys

from conf import N_EM_GENS, N_MAIN_GENS # For error exit

# --- Ensure local imports work ---
# Add other necessary imports if they are missing or causing issues
try:
    from grid_env import PowerGridEnv
    from sarsa_agent import SARSAgent # Import the agent
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure grid_env.py and sarsa_agent.py are in the same directory or accessible in the Python path.")
    sys.exit(1)
# --- End Local Import Handling ---

def run_training(env, agent, episodes, save_file):
    """Runs the SARSA training loop."""
    print(f"Starting SARSA training for {episodes} episodes...")
    total_rewards = []
    # Frequency for progress reporting and saving
    report_interval = max(1, episodes // 50) # Report ~50 times
    save_interval = max(10, episodes // 10) # Save ~10 times + final

    for episode in tqdm(range(episodes), desc="Training Progress"):
        obs, _ = env.reset()
        # Ensure observation is valid before discretizing
        if obs is None:
             tqdm.write(f"ERROR: env.reset() returned None observation at episode {episode+1}. Aborting.")
             break # Or handle appropriately
        try:
            state = agent.discretize_state(obs)
        except Exception as e:
            tqdm.write(f"ERROR: Failed to discretize initial state at episode {episode+1}. Obs: {obs}. Error: {e}")
            break # Or handle appropriately

        # Initial action a
        action_tuple = agent.choose_action(state)
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            step_count += 1
            # Take action a, observe r, s'
            try:
                next_obs, reward, terminated, truncated, info = env.step(action_tuple)
            except Exception as e:
                 tqdm.write(f"\nERROR: env.step() failed at episode {episode+1}, step {step_count}. Action: {action_tuple}. Error: {e}")
                 # Optionally try to recover or just end the episode/training
                 done = True # End this episode
                 reward = -1000 # Penalize heavily for env error?
                 next_obs = obs # Keep previous observation to avoid crashing discretization
                 terminated = True # Mark as terminated to exit loop

            # Ensure next_obs is valid
            if next_obs is None:
                 tqdm.write(f"\nERROR: env.step() returned None observation at episode {episode+1}, step {step_count}. Aborting episode.")
                 done = True
                 next_obs = obs # Use previous obs to avoid crash

            # Discretize next state
            try:
                 next_state = agent.discretize_state(next_obs)
            except Exception as e:
                tqdm.write(f"\nERROR: Failed to discretize next state at episode {episode+1}, step {step_count}. Next Obs: {next_obs}. Error: {e}")
                done = True # End episode
                next_state = state # Use previous state to avoid crash

            # Choose next action a' from s' using policy (even if done, for final update)
            next_action_tuple = agent.choose_action(next_state)

            # Update Q(s,a) using s, a, r, s', a'
            # Only update if the episode didn't end due to an error before getting valid next_state/action
            if not (done and next_state == state): # Avoid update if state didn't progress meaningfully due to error
                 agent.update(state, action_tuple, reward, next_state, next_action_tuple)

            # Update state and action for the next iteration
            state = next_state
            action_tuple = next_action_tuple
            episode_reward += reward

            # Check Gymnasium V26 termination/truncation
            if terminated or truncated:
                 done = True

        # End of episode
        total_rewards.append(episode_reward)

        # Optional: Print progress periodically
        if (episode + 1) % report_interval == 0 or episode == episodes - 1:
            avg_reward = np.mean(total_rewards[-report_interval:])
            tqdm.write(f"Episode {episode+1}/{episodes} | Avg Reward (last {report_interval}): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}") # Keep Epsilon print

        # Save intermediate Q-table periodically
        if (episode + 1) % save_interval == 0:
             agent.save_q_table(save_file)

    print("Training finished.")
    agent.save_q_table(save_file) # Final save


def run_demonstration(env, agent, demo_steps, render_mode):
    """Runs the agent in demonstration mode with greedy actions."""
    print(f"Starting demonstration for {demo_steps} steps...")
    agent.epsilon = 0.0 # Ensure greedy actions

    try:
        obs, _ = env.reset()
        total_reward = 0
        steps_taken = 0

        # Check initial observation
        if obs is None:
             print("ERROR: env.reset() returned None observation at start of demonstration.")
             return

        for step in range(demo_steps):
            try:
                state = agent.discretize_state(obs)
            except Exception as e:
                print(f"\nERROR: Failed to discretize state at demo step {steps_taken+1}. Obs: {obs}. Error: {e}")
                break

            action_tuple = agent.choose_action(state) # Greedy action

            try:
                obs, reward, terminated, truncated, info = env.step(action_tuple)
                # Rendering is now handled within env.step based on env.render_mode
            except Exception as e:
                 print(f"\nERROR: env.step() failed at demo step {steps_taken+1}. Action: {action_tuple}. Error: {e}")
                 break # Stop demonstration on error

            total_reward += reward
            steps_taken += 1
            # --- Terminal Rendering Logic Moved Here ---
            if render_mode == 'terminal':
                if not info:
                    print(f"\n--- Step {steps_taken} ---")
                    print("  No info dictionary available for rendering.")
                else:
                    # Calculate step number based on steps_taken for clarity
                    step_number = steps_taken
                    print(f"\n--- Step {step_number} ---")

                    action_str = str(action_tuple) if action_tuple is not None else "N/A"
                    reward_str = f"{reward:.2f}" if reward is not None else "N/A" # Use current step reward
                    print(f"Action Taken: {action_str}")
                    print(f"Step Reward: {reward_str}")

                    # Get values and ensure they are floats before formatting
                    current_time_val = info.get('current_time', 0.0)
                    deficit_val = info.get('power_balance_deficit_MW', 0.0)

                    try:
                        print(f"  Current Time: {float(current_time_val):.2f} hrs")
                        print(f"  Deficit: {float(deficit_val):.2f} MW")
                    except (ValueError, TypeError) as e:
                        print(f"  ERROR rendering numeric data: {e}")

                    # Use N_MAIN_GENS and N_EM_GENS imported from conf
                    main_online_count = sum(info.get('main_gen_online', []))
                    em_online_count = sum(info.get('emergency_gens_online', []))
                    print(f"  Main Gens Online: {main_online_count}/{N_MAIN_GENS}")
                    print(f"  EM Gens Online: {em_online_count}/{N_EM_GENS}")
                    if info.get('critical_failure', False):
                        print("  [CRITICAL FAILURE] High priority load not met!")
            # --- End of Terminal Rendering Logic ---

            # Check for None observation after step
            if obs is None:
                 print(f"\nERROR: env.step() returned None observation at demo step {steps_taken}.")
                 break

            if terminated or truncated:
                print(f"\nEpisode finished within demo steps (at step {steps_taken}). Resetting env.")
                obs, _ = env.reset()
                if obs is None:
                    print("\nERROR: env.reset() returned None observation after episode finish.")
                    break
                if steps_taken >= demo_steps: # Exit if requested steps reached
                    break
                # No explicit sleep needed here, env.render handles it for 'human' mode

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    finally:
        print(f"\nDemonstration finished after {steps_taken} steps.")
        print(f"Final Total Reward during demo: {total_reward:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or run SARSA agent for PowerGridEnv.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'demo'],
                        help="Mode to run: 'train' or 'demo'")
    parser.add_argument('--episodes', type=int, default=5000,
                        help="Number of episodes for training.")
    parser.add_argument('--demosteps', type=int, default=50, # Increased default demo steps
                        help="Maximum number of steps for demonstration.")
    parser.add_argument('--load', action='store_true',
                        help="Load pre-trained Q-table before starting.")
    parser.add_argument('--save_file', type=str, default="sarsa_q_table.pkl",
                        help="Filename for saving/loading the Q-table.")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate (alpha)")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Constant epsilon for exploration")
    # Render mode argument for environment initialization
    parser.add_argument('--render_mode', type=str, choices=['human', 'terminal', 'none'], default='human',
                        help="Render mode for demonstration: 'human' (clears terminal), 'terminal' (prints summary), or 'none'.")

    args = parser.parse_args()

    # --- Environment Setup ---
    # Pass the requested render mode from args, but only for demo mode
    # For training, render_mode should be None/None so render() inside step does nothing
    render_mode_for_env = 'human' if args.mode == 'demo' and args.render_mode == 'human' else None
    try:
        env = PowerGridEnv(render_mode=render_mode_for_env)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        sys.exit(1)


    # --- Agent Setup ---
    # Note: Discretization bins might need tuning! See sarsa_agent.py comments.
    try:
        agent = SARSAgent(env,
                          learning_rate=args.lr,
                          discount_factor=args.gamma,
                          epsilon=args.epsilon) # Constant epsilon
    except Exception as e:
        print(f"Error initializing SARSA agent: {e}")
        env.close()
        sys.exit(1)

    # --- Load Q-table if requested ---
    q_table_loaded = False
    if args.load:
        print(f"Attempting to load Q-table from: {args.save_file}")
        q_table_loaded = agent.load_q_table(args.save_file)
        if q_table_loaded:
            print("Q-table loaded successfully.")
        elif args.mode == 'demo':
             print(f"ERROR: Failed to load Q-table '{args.save_file}', which is required for demo mode.")
             env.close()
             sys.exit(1)
        elif args.mode == 'train':
             print(f"WARN: Failed to load Q-table '{args.save_file}'. Starting training from scratch.")

    # --- Check prerequisites for demo mode ---
    if args.mode == 'demo':
        if not args.load: # Check --load explicitly
             print("ERROR: Demo mode requires loading a Q-table. Use the --load flag.")
             env.close()
             sys.exit(1)
        if not q_table_loaded:
             print("ERROR: Cannot run demo because Q-table failed to load.")
             env.close()
             sys.exit(1)

    # --- Run Mode ---
    try:
        if args.mode == 'train':
            run_training(env, agent, args.episodes, args.save_file)
        elif args.mode == 'demo':
            # run_demonstration already checked for load success
            run_demonstration(env, agent, args.demosteps, args.render_mode)
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
    finally:
        # --- Cleanup ---
        print("Closing environment...")
        env.close()
        print("Execution finished.")