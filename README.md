# Power Grid Reinforcement Learning Environment with SARSA Agent

This project implements a simplified power grid simulation environment using Gymnasium and trains a SARSA (State-Action-Reward-State-Action) agent to manage grid operations. The agent learns to control battery charging/discharging, deploy emergency generators, and shed low-priority loads to maintain power balance, meet demand based on priority, and minimize operational costs and penalties over a simulated period.

**Core Components:**

*   **Environment (`grid_env.py`):** Simulates the power grid dynamics based on `Gymnasium`. Includes main/emergency generators, battery storage, and prioritized load zones with dynamic demand (`utilites.py`, `conf.py`).
*   **Agent (`sarsa_agent.py`):** Implements the SARSA algorithm with constant epsilon-greedy exploration and state discretization to handle the environment's continuous features. Learns a Q-table mapping discretized states to action values.
*   **Main Script (`main.py`):** Provides a command-line interface to train the agent or run demonstrations with a trained agent.
*   **Configuration (`conf.py`):** Centralizes constants for environment parameters (capacities, rates, probabilities, timings) and reward function weights/costs.
*   **Utilities (`utilites.py`):** Contains helper classes for generators, battery, load zones, and the demand profile function.

## 1. Prerequisites

*   **Python:** Version 3.8 or higher recommended.
*   **Libraries:** Install required libraries using pip:
    ```bash
    pip install numpy tqdm gymnasium
    ```
*   **Files:** Ensure all project files are in the **same directory**:
    *   `main.py`
    *   `sarsa_agent.py`
    *   `grid_env.py`
    *   `utilites.py`
    *   `conf.py`
    *   (Optional but helpful: `README.md`, `Power Grid RL Environment Design.md`)

## 2. Running the Code

Execute the simulation from your terminal or command prompt.

*   **Navigate to Directory:** Open your terminal and use the `cd` command to navigate to the directory containing the Python files.
    ```bash
    cd path/to/your/project_directory
    ```
*   **Execute `main.py`:** Use `python main.py` followed by arguments to specify the desired mode and parameters.

## 3. Training the Agent (`--mode train`)

This mode runs the SARSA algorithm. The agent interacts with the environment over many episodes to learn a Q-table (representing its policy). The Q-table is periodically saved. Rendering is disabled during training for performance.

*   **Basic Training:**
    *   Starts training with default settings (5000 episodes, LR=0.1, Gamma=0.99, Epsilon=0.1).
    *   Saves the learned Q-table to `sarsa_q_table.pkl`.
    ```bash
    python main.py --mode train
    ```

*   **Specify Episode Count:**
    *   Train for a different number of episodes (e.g., 20000). More episodes generally improve learning but increase training time.
    ```bash
    python main.py --mode train --episodes 20000
    ```

*   **Continue Training (Load Existing Q-table):**
    *   Use the `--load` flag to load an existing Q-table (from `--save_file`, default `sarsa_q_table.pkl`) and continue training.
    ```bash
    python main.py --mode train --episodes 10000 --load
    ```
    *   *Note:* If loading fails, training will start from scratch with a warning.

*   **Adjust Hyperparameters:**
    *   Experiment with learning rate (`--lr`), discount factor (`--gamma`), and the *constant* exploration rate (`--epsilon`). Finding good hyperparameters often requires experimentation.
    ```bash
    python main.py --mode train --episodes 10000 --lr 0.05 --gamma 0.98 --epsilon 0.2
    ```

*   **Specify Save File:**
    *   Save the Q-table to a custom filename. This file will also be used by `--load` if specified.
    ```bash
    python main.py --mode train --save_file my_trained_agent_v1.pkl
    ```

**During Training:**

*   A `tqdm` progress bar shows episode completion.
*   Periodic updates display the average reward over recent episodes and the current epsilon value.
*   Messages indicate when the Q-table is being saved.

## 4. Running a Demonstration (`--mode demo`)

This mode loads a previously trained Q-table (**`--load` flag is mandatory**) and runs the agent in the environment using the learned policy (acting greedily, epsilon=0). The environment's state can be visualized at each step.

*   **Requirement:** You **must** use `--load` and have a valid Q-table file (specified by `--save_file`, default `sarsa_q_table.pkl`).

*   **Rendering Modes (`--render_mode`):**
    *   `human` (Default): Clears the terminal at each step and displays a formatted "Power Grid Dashboard" showing the full system state (power balance, loads, battery, generators). Pauses briefly between steps for visibility.
    *   `terminal`: Prints a summary of each step (action, reward, key states) sequentially to the terminal without clearing the screen. Useful for logging or quick inspection.
    *   `none`: Runs the demonstration without any per-step visual output. Only the final summary message (total steps, total reward) is printed.

*   **Examples:**

    *   **Basic Demo (Human Render):** Load `sarsa_q_table.pkl`, run for default 50 steps, render the dashboard.
        ```bash
        python main.py --mode demo --load
        ```
        *(Equivalent to: `python main.py --mode demo --load --render_mode human --demosteps 50`)*

    *   **Terminal Render Demo:** Load `sarsa_q_table.pkl`, run for 24 steps, print summary each step.
        ```bash
        python main.py --mode demo --load --render_mode terminal --demosteps 24
        ```

    *   **No Render Demo:** Load `sarsa_q_table.pkl`, run for 100 steps without intermediate output.
        ```bash
        python main.py --mode demo --load --render_mode none --demosteps 100
        ```

    *   **Load Specific Q-table (Human Render):** Load a custom Q-table file, run for 48 steps.
        ```bash
        python main.py --mode demo --load --save_file my_trained_agent_v1.pkl --render_mode human --demosteps 48
        ```

**During Demonstration:**

*   Output corresponding to the chosen `--render_mode` is displayed.
*   A message indicates if an episode finishes before the requested `--demosteps` are completed.
*   A final summary shows the total steps taken and the cumulative reward achieved during the demonstration run.

## 5. Important Note on State Discretization

The performance of the SARSA agent heavily depends on how the continuous or large-range parts of the environment's state (like time, battery SoC, load demands, generator timers/runtime) are **discretized** into bins. This mapping happens in the `SARSAgent.discretize_state` method in `sarsa_agent.py`.

*   **Default Bins:** The agent uses a default binning strategy defined in `SARSAgent.__init__`. **This default is a starting point and likely requires tuning for optimal performance.**
*   **Tuning:** If the agent struggles to learn (e.g., low rewards, poor decisions in demo), experiment with:
    *   The number of bins for each feature (`num_bins` tuple in `SARSAgent.__init__`).
    *   The boundaries used for binning continuous values (e.g., `load_demand_bins`, `em_runtime_max` in `SARSAgent.__init__`).
*   **Trade-offs:**
    *   *Fewer bins:* Smaller state space, faster learning, but might group dissimilar states, hindering optimal policy discovery.
    *   *More bins:* Larger state space, potentially better policy representation, but significantly slower learning and requires more data (episodes). A very large state space can become computationally infeasible ("curse of dimensionality").

Finding the right discretization is often a key part of applying Q-learning or SARSA to environments with continuous features.
