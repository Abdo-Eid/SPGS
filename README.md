Okay, here's how to run the code you have set up:

**1. Prerequisites:**

*   **Python:** Make sure you have Python 3 installed.
*   **Libraries:** You need `numpy`, `tqdm`, and `gymnasium`. Install them if you haven't already:
    ```bash
    pip install numpy tqdm gymnasium
    ```
*   **Files:** Ensure all the Python files are in the **same directory**:
    *   `main.py`
    *   `sarsa_agent.py`
    *   `grid_env.py`
    *   `utilites.py`
    *   `conf.py`

**2. Running the Code:**

You will run the code from your terminal or command prompt.

*   **Navigate to Directory:** Open your terminal and use the `cd` command to go to the directory where you saved the Python files. For example:
    ```bash
    cd path/to/your/project_directory
    ```
*   **Execute `main.py`:** You will use the `python main.py` command followed by arguments to specify whether you want to train or run a demonstration.

**3. Training the Agent (`--mode train`)**

This mode runs the SARSA algorithm to learn a Q-table (the agent's "brain") by interacting with the environment over many episodes.

*   **Basic Training:** Start training with default settings (5000 episodes, default learning rate, etc.). The Q-table will be saved as `sarsa_q_table.pkl`.
    ```bash
    python main.py --mode train
    ```
*   **Specify Number of Episodes:** Train for a different number of episodes (e.g., 10000). More episodes usually lead to better learning but take longer.
    ```bash
    python main.py --mode train --episodes 10000
    ```
*   **Continue Training (Load Existing Q-table):** If you previously trained and saved a `sarsa_q_table.pkl` file (or specified a different `--save_file`), you can continue training from it using the `--load` flag.
    ```bash
    python main.py --mode train --episodes 5000 --load
    ```
*   **Adjust Hyperparameters:** You can experiment with the learning rate (`--lr`), discount factor (`--gamma`), and the constant exploration rate (`--epsilon`).
    ```bash
    python main.py --mode train --episodes 10000 --lr 0.05 --gamma 0.98 --epsilon 0.2
    ```
*   **Specify Save File:** Save the Q-table to a different file name.
    ```bash
    python main.py --mode train --save_file my_trained_agent.pkl
    ```

    **During Training, you will see:**
    *   A progress bar showing the episodes completed.
    *   Periodic updates on the average reward over the last 100 episodes.
    *   Messages indicating when the Q-table is being saved.

**4. Running a Demonstration (`--mode demo`)**

This mode loads a previously trained Q-table and runs the agent in the environment using the learned policy (choosing the best actions). It will render the environment state step-by-step in your terminal.

*   **Basic Demonstration:** Load the default `sarsa_q_table.pkl` and run for 100 steps. **`--load` is required.**
    ```bash
    python main.py --mode demo --load
    ```
*   **Specify Demo Length:** Run the demonstration for a different number of steps (e.g., 200).
    ```bash
    python main.py --mode demo --load --demosteps 200
    ```
*   **Load Specific Q-table:** If you saved your trained agent to a different file during training, specify it here.
    ```bash
    python main.py --mode demo --load --save_file my_trained_agent.pkl
    ```

    **During Demonstration, you will see:**
    *   The terminal clearing and updating at each step (due to the `render()` function).
    *   The "Power Grid Dashboard" showing the current state (time, power balance, load status, battery, generators).
    *   Information printed below the dashboard for each step: the action taken, the reward received, the cumulative reward, and critical failure warnings.

**Important Note on Discretization:**

Remember that the performance of the SARSA agent heavily depends on how the continuous parts of the state (like time, battery SoC, load demands, generator timers/runtime) are discretized into bins in `sarsa_agent.py`. The default settings are just a starting point. If the agent doesn't learn well, you may need to experiment with the `num_bins` defined within the `SARSAgent.__init__` method.