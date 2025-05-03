**How to Run:**

1.  **Save:** Save the code above as a Python file, for example, `power_grid_sarsa.py`.
2.  **Install Libraries:** Make sure you have the necessary libraries.
    ```bash
    pip install networkx matplotlib gymnasium numpy
    ```
    (You no longer need `stable-baselines3` or `torch`/`tensorflow` for this SARSA version).
3.  **Train the SARSA Agent:**
    ```bash
    python power_grid_sarsa.py --mode train --episodes 10000
    ```
    *   `--mode train`: Tells the script to run in training mode.
    *   `--episodes 10000`: Specifies the number of training episodes. More episodes generally lead to a better-trained agent, but training time increases. Start with a few thousand and increase if needed.
    *   You can add `--load` to continue training from a previously saved Q-table if the file exists.
    *   The Q-table will be saved to `power_grid_sarsa_qtable.npy` by default after training.

4.  **Run the Demonstration:**
    ```bash
    python power_grid_sarsa.py --mode demo --demosteps 100 --load
    ```
    *   `--mode demo`: Tells the script to run in demonstration mode.
    *   `--demosteps 100`: Specifies how many simulation steps to run the demonstration for.
    *   `--load`: **Required** in demo mode to load the trained Q-table from the `.npy` file.
    *   A Matplotlib window will pop up showing the grid state as the agent takes actions.
