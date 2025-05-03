"""
SARSA Agent Implementation.

This module defines the `SARSAgent` class, which implements the State-Action-Reward-State-Action
Reinforcement Learning algorithm. This version uses:
- A constant epsilon for exploration (epsilon-greedy policy).
- Discretization for handling potentially continuous observation spaces.
- A Q-table stored as a defaultdict to manage learned state-action values.
- Support for Gymnasium's `MultiDiscrete` action spaces.
"""

import numpy as np
import pickle
from collections import defaultdict
import gymnasium as gym # Used for action space type checking and structure

class SARSAgent:
    """
    A SARSA agent with constant epsilon-greedy exploration and state discretization.

    Manages a Q-table to learn optimal actions in an environment. It requires
    the environment to have a `MultiDiscrete` action space and provides methods
    for choosing actions, updating the Q-table based on experience (s, a, r, s', a'),
    and saving/loading the learned Q-table. State discretization is handled internally
    based on predefined or default binning strategies.

    Attributes:
        env: The Gymnasium environment instance.
        lr (float): Learning rate (alpha) for Q-table updates.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Constant probability for choosing a random action (exploration).
        action_nvec (np.ndarray): Vector containing the number of discrete actions for each component
                                  of the MultiDiscrete action space.
        num_actions (int): The total number of unique action combinations.
        q_table (defaultdict): The Q-table storing state-action values. Maps state tuples
                               to NumPy arrays of Q-values for each action index.
        num_bins (tuple): Tuple defining the number of bins for each dimension of the
                          discretized state space.
        # Attributes related to discretization boundaries (potentially set in __init__):
        load_demand_bins (list): List of tuples defining (min, max) ranges for load demand discretization.
        em_runtime_max (int): Maximum runtime steps for emergency generators, used for binning.
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, # Adjusted default gamma
                 epsilon=0.1, num_bins_per_feature=None):
        """
        Initializes the SARSA Agent.

        Args:
            env: The Gymnasium environment instance. Must have a MultiDiscrete action space
                 and a Box observation space (or one compatible with the discretization logic).
            learning_rate (float): The learning rate (alpha) for updates.
            discount_factor (float): The discount factor (gamma) for future rewards.
            epsilon (float): The constant probability for exploration (epsilon-greedy).
            num_bins_per_feature (tuple | list | None): A tuple defining the number of bins
                for each feature in the observation space that needs discretization.
                The length and order must match the structure defined in `discretize_state`.
                If None, a default binning strategy is attempted (requires careful review!).
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon # Constant exploration rate

        # --- Action Space Handling ---
        if not isinstance(env.action_space, gym.spaces.MultiDiscrete):
            raise ValueError("SARSAgent currently requires a MultiDiscrete action space.")
        self.action_nvec = env.action_space.nvec # e.g., [3, 3, 3, 2, 2]
        # Total number of unique actions = product of options for each component
        self.num_actions = np.prod(self.action_nvec).item()

        # --- State Space Discretization Setup ---
        # Get observation space bounds (may be -inf/inf, handle in discretize_state)
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high

        # --- IMPORTANT: Define Discretization Bins ---
        # This section defines how continuous (or large-range integer) features
        # in the observation space are mapped to discrete bins.
        # The structure MUST match the observation vector created by env._get_obs().
        if num_bins_per_feature is None:
            # --- Default Binning Strategy (NEEDS CAREFUL TUNING!) ---
            print("WARN: Using default discretization bins. Review and tune for optimal performance!")
            # Observation structure from grid_env._get_obs():
            # [time_norm, batt_soc_norm, batt_mode,
            #  load_hi_demand, load_hi_shed, load_lo1_demand, load_lo1_shed, load_lo2_demand, load_lo2_shed,
            #  main_gen1..., main_gen2..., main_gen3...,
            #  em_gen1..., em_gen2..., em_gen3...]

            bins = []
            # 1. Time (normalized 0-1): 1 feature
            bins.append(10) # 10 bins for time

            # 2. Battery (normalized SoC 0-1, mode 0/1/2): 2 features
            bins.extend([10, 3]) # 10 bins for SoC, 3 bins (exact) for mode

            # 3. Load Zones (demand, shed status per zone): 2 features * num_zones
            # Define approximate expected ranges for demand (can be refined by environment analysis)
            # Max demand from profile * scale * 100: Hi=1.0*1.5*100=150, Lo1=0.7*1.5*100=105, Lo2=0.5*1.5*100=75
            # Add some buffer.
            self.load_demand_bins = [(0, 160), (0, 110), (0, 80)] # (min, max) ranges for hi, lo1, lo2
            num_demand_bins = 10 # Use 10 bins for demand values
            bins.extend([num_demand_bins, 1]) # Hi Demand (10 bins), Hi Shed (1 bin - always 0)
            bins.extend([num_demand_bins, 2]) # Lo1 Demand(10 bins), Lo1 Shed (2 bins - 0/1)
            bins.extend([num_demand_bins, 2]) # Lo2 Demand(10 bins), Lo2 Shed (2 bins - 0/1)

            # 4. Main Generators (online, fail_timer, min_out, max_out): 4 features * N_MAIN_GENS
            # Need heal_time_steps from an environment instance
            if not env.main_generators: raise ValueError("Environment has no main generators for binning.")
            main_heal_bins = env.main_generators[0].heal_time_steps + 1 # Bins: 0, 1, ..., heal_time
            for _ in range(len(env.main_generators)):
                 # Online (2 bins: 0/1), Timer (heal_bins), MinOut (1 bin: ignored), MaxOut (1 bin: ignored)
                 bins.extend([2, main_heal_bins, 1, 1])

            # 5. Emergency Generators (online, start_timer, runtime_left): 3 features * N_EM_GENS
            if not env.emergency_generators: raise ValueError("Environment has no emergency generators for binning.")
            em_boot_bins = env.emergency_generators[0].boot_time_steps + 1 # Bins: 0, 1, ..., boot_time
            em_runtime_bins = 10 # Bin remaining runtime into 10 levels (e.g., 0-10%, 10-20%, ...)
            self.em_runtime_max = env.emergency_generators[0].total_runtime_steps # Get max runtime for scaling
            for _ in range(len(env.emergency_generators)):
                # Online (2 bins: 0/1), Timer (boot_bins), Runtime (runtime_bins)
                bins.extend([2, em_boot_bins, em_runtime_bins])

            self.num_bins = tuple(bins)
            print(f"Using default discretization bins: {self.num_bins}")
            # Calculate estimated state space size (can be very large!)
            estimated_states = np.prod([b for b in self.num_bins if b > 0], dtype=np.float64) # Avoid overflow
            print(f"Estimated number of discrete states: {estimated_states:.2e}")
            if estimated_states > 1e9:
                 print("WARNING: State space is very large, learning may be slow or infeasible.")
        else:
            # Use user-provided bins
            self.num_bins = tuple(num_bins_per_feature)
            # Note: If using custom bins, ensure helper attributes like
            # self.load_demand_bins and self.em_runtime_max are set appropriately
            # if the default logic above wasn't executed or needs overriding.
            # This might require adding parameters or logic here. For now, assume
            # the discretization function can handle it or defaults are sufficient.
            print(f"Using provided discretization bins: {self.num_bins}")


        # Verify bin configuration matches observation space dimension
        if len(self.num_bins) != env.observation_space.shape[0]:
             raise ValueError(f"Length of num_bins ({len(self.num_bins)}) must match "
                              f"observation space dimension ({env.observation_space.shape[0]})")

        # --- Q-Table Initialization ---
        # Use defaultdict: if a state is visited for the first time, it automatically
        # gets an entry with an array of zeros (one Q-value per action).
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))


    def _action_tuple_to_index(self, action_tuple):
        """
        Maps a MultiDiscrete action tuple (e.g., (1, 0, 2, 1, 0)) to a single integer index.

        This is necessary because the Q-table stores values in a 1D array per state,
        indexed by this single action index. The mapping treats the tuple as digits
        in a mixed radix number system.

        Args:
            action_tuple (tuple or list): The action tuple from the agent or environment.
                                          Length must match `len(self.action_nvec)`.

        Returns:
            int: The single integer index representing this action combination.

        Raises:
            ValueError: If the action tuple length or element values are invalid.
        """
        if len(action_tuple) != len(self.action_nvec):
            raise ValueError(f"Action tuple length mismatch. Expected {len(self.action_nvec)}, got {len(action_tuple)}.")

        index = 0
        factor = 1
        # Iterate backwards through the action tuple and nvec
        for i in range(len(self.action_nvec) - 1, -1, -1):
            action_component = action_tuple[i]
            num_options = self.action_nvec[i]
            # Check bounds for this action component
            if not (0 <= action_component < num_options):
                 raise ValueError(f"Action element {action_component} at index {i} is out of bounds [0, {num_options}). Action: {action_tuple}")
            # Add the component's contribution to the index
            index += action_component * factor
            # Update the factor for the next (more significant) position
            factor *= num_options

        return int(index) # Ensure it's a standard Python int


    def _action_index_to_tuple(self, index):
        """
        Maps a single integer action index back to a MultiDiscrete action tuple.

        This is the inverse of `_action_tuple_to_index`. Used when the agent selects
        an action index (e.g., greedily from Q-table) and needs to convert it back
        to the tuple format expected by the environment's `step` method.

        Args:
            index (int): The single integer index representing an action combination.
                         Must be in the range [0, self.num_actions).

        Returns:
            tuple: The MultiDiscrete action tuple corresponding to the index.

        Raises:
            ValueError: If the index is out of bounds.
        """
        if not (0 <= index < self.num_actions):
            raise ValueError(f"Action index {index} out of bounds [0, {self.num_actions}).")

        action_tuple = [0] * len(self.action_nvec)
        remainder = index

        # Iterate forwards through the action components
        for i in range(len(self.action_nvec)):
            # Calculate the factor (product of sizes of remaining components)
            factor = 1
            if i < len(self.action_nvec) - 1:
                 # Use np.prod for safety with potentially large numbers
                 factor = np.prod(self.action_nvec[i+1:], dtype=np.int64).item()

            # Determine the value for this action component
            action_val = remainder // factor
            action_tuple[i] = action_val

            # Update the remainder for the next component
            remainder %= factor

        return tuple(action_tuple)

    def discretize_state(self, observation):
        """
        Discretizes a continuous or mixed observation vector into a tuple of bin indices.

        This function takes the raw observation array from the environment and maps
        each feature to a discrete bin index based on the `self.num_bins` configuration
        and potentially defined ranges (like `self.load_demand_bins`). The resulting
        tuple serves as the key for the Q-table.

        Args:
            observation (np.ndarray): The observation vector from the environment.
                                      Must match the structure expected by the binning logic.

        Returns:
            tuple: A tuple of integer bin indices representing the discretized state.

        Raises:
            ValueError: If the observation length doesn't match expected dimension.
            IndexError: If binning logic accesses invalid observation indices.
        """
        if len(observation) != len(self.num_bins):
             raise ValueError(f"Observation length ({len(observation)}) does not match "
                              f"number of bin definitions ({len(self.num_bins)}).")

        state = [0] * len(self.num_bins) # Initialize state tuple elements
        obs_idx = 0 # Track current position in the raw observation vector

        try:
            # 1. Time (1 feature, normalized 0-1)
            time_val = np.clip(observation[obs_idx], 0, 1) # Ensure value is within [0, 1]
            # np.linspace creates edges; endpoint=False means last edge is not included, matching digitize behavior
            time_bins = np.linspace(0, 1, self.num_bins[obs_idx], endpoint=False)
            # np.digitize returns indices starting from 1; subtract 1 for 0-based indexing
            state[obs_idx] = np.digitize(time_val, time_bins) - 1
            obs_idx += 1

            # 2. Battery (2 features: soc_norm 0-1, mode 0/1/2)
            # SoC (normalized 0-1)
            soc_val = np.clip(observation[obs_idx], 0, 1)
            soc_bins = np.linspace(0, 1, self.num_bins[obs_idx], endpoint=False)
            state[obs_idx] = np.digitize(soc_val, soc_bins) - 1
            obs_idx += 1
            # Mode (already discrete 0, 1, 2) - directly use the value if num_bins is 3
            batt_mode = int(observation[obs_idx])
            if self.num_bins[obs_idx] == 3:
                 state[obs_idx] = np.clip(batt_mode, 0, 2) # Ensure valid mode index
            else:
                 # Handle potential mismatch if num_bins wasn't set to 3
                 state[obs_idx] = np.digitize(batt_mode, np.linspace(0, 2, self.num_bins[obs_idx], endpoint=False)) -1
            obs_idx += 1

            # 3. Load Zones (2 features per zone: demand, shed_status)
            load_zone_keys = ['hi', 'lo1', 'lo2'] # Order must match _get_obs()
            for i, zone_key in enumerate(load_zone_keys):
                 # Demand (continuous, needs binning based on defined ranges)
                 demand_val = observation[obs_idx]
                 min_demand, max_demand = self.load_demand_bins[i] # Get range for this zone
                 # Clip value to the defined range before binning
                 demand_val_clipped = np.clip(demand_val, min_demand, max_demand)
                 # Create bins within the range
                 demand_bins = np.linspace(min_demand, max_demand, self.num_bins[obs_idx], endpoint=False)
                 state[obs_idx] = np.digitize(demand_val_clipped, demand_bins) - 1
                 obs_idx += 1

                 # Shed Status (already discrete 0/1)
                 shed_status = int(observation[obs_idx])
                 # If num_bins is 1 (e.g., for 'hi' zone), index is always 0
                 # If num_bins is 2 (e.g., for 'lo' zones), index is 0 or 1
                 if self.num_bins[obs_idx] == 1:
                      state[obs_idx] = 0
                 elif self.num_bins[obs_idx] == 2:
                      state[obs_idx] = np.clip(shed_status, 0, 1)
                 else: # Handle potential mismatch
                      state[obs_idx] = np.digitize(shed_status, np.linspace(0, 1, self.num_bins[obs_idx], endpoint=False)) - 1
                 obs_idx += 1

            # 4. Main Generators (4 features per gen: online, fail_timer, min_out, max_out)
            for i in range(len(self.env.main_generators)):
                 # Online status (discrete 0/1)
                 online_status = int(observation[obs_idx])
                 state[obs_idx] = np.clip(online_status, 0, 1) # Bin index 0 or 1
                 obs_idx += 1

                 # Fail timer (integer 0 to heal_time_steps)
                 timer_val = int(observation[obs_idx])
                 max_timer = self.env.main_generators[i].heal_time_steps
                 # Number of bins should be max_timer + 1 (for 0)
                 # Use value directly as bin index after clipping
                 state[obs_idx] = np.clip(timer_val, 0, max_timer)
                 obs_idx += 1

                 # Min Output (ignored - 1 bin)
                 state[obs_idx] = 0
                 obs_idx += 1
                 # Max Output (ignored - 1 bin)
                 state[obs_idx] = 0
                 obs_idx += 1

            # 5. Emergency Generators (3 features per gen: online, start_timer, runtime_left)
            for i in range(len(self.env.emergency_generators)):
                # Online status (discrete 0/1)
                online_status = int(observation[obs_idx])
                state[obs_idx] = np.clip(online_status, 0, 1)
                obs_idx += 1

                # Start timer (integer 0 to boot_time_steps)
                timer_val = int(observation[obs_idx])
                max_timer = self.env.emergency_generators[i].boot_time_steps
                # Use value directly as bin index after clipping
                state[obs_idx] = np.clip(timer_val, 0, max_timer)
                obs_idx += 1

                # Runtime left (continuous/large integer, needs binning)
                runtime_val = observation[obs_idx]
                # Clip to [0, max_runtime]
                runtime_val_clipped = np.clip(runtime_val, 0, self.em_runtime_max)
                # Create bins from 0 to max_runtime
                runtime_bins = np.linspace(0, self.em_runtime_max, self.num_bins[obs_idx], endpoint=False)
                state[obs_idx] = np.digitize(runtime_val_clipped, runtime_bins) - 1
                obs_idx += 1

        except IndexError:
             print(f"ERROR: IndexError during discretization at obs_idx={obs_idx}.")
             print(f"Observation length: {len(observation)}, Num bins defined: {len(self.num_bins)}")
             raise # Re-raise the error after printing info

        # Final check: Ensure all bin indices are non-negative after the '- 1' adjustments
        state = [max(0, s) for s in state]

        return tuple(state) # Q-table key must be hashable, so use a tuple


    def choose_action(self, state_tuple):
        """
        Chooses an action using the epsilon-greedy policy based on the current Q-table.

        With probability epsilon, selects a random action (exploration).
        With probability 1-epsilon, selects the action with the highest Q-value
        for the given state (exploitation). Ties are broken randomly.

        Args:
            state_tuple (tuple): The discretized state tuple (key for the Q-table).

        Returns:
            tuple: The chosen action tuple in the format expected by the environment.
        """
        # Epsilon-greedy decision
        if np.random.rand() < self.epsilon:
            # --- Explore ---
            # Choose a random action *index* uniformly from all possible actions
            action_index = np.random.randint(self.num_actions)
        else:
            # --- Exploit ---
            # Get the Q-values for the current state from the table
            q_values = self.q_table[state_tuple] # defaultdict handles unseen states

            # Find the index (or indices) of the action(s) with the highest Q-value
            max_q = np.max(q_values)
            # Get all indices where the Q-value equals the maximum
            best_action_indices = np.where(q_values == max_q)[0]

            # Choose randomly among the best actions (handles ties)
            action_index = np.random.choice(best_action_indices)

        # Convert the chosen single action index back to the MultiDiscrete action tuple
        action_tuple = self._action_index_to_tuple(action_index)
        return action_tuple

    def update(self, state_tuple, action_tuple, reward, next_state_tuple, next_action_tuple):
        """
        Updates the Q-table using the SARSA update rule.

        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * Q(s', a') - Q(s, a)]

        Where:
        s = state_tuple
        a = action_tuple
        r = reward
        s' = next_state_tuple
        a' = next_action_tuple (the action actually taken in the next state)

        Args:
            state_tuple (tuple): The discretized state the action was taken from.
            action_tuple (tuple): The action tuple that was taken.
            reward (float): The reward received after taking the action.
            next_state_tuple (tuple): The resulting discretized state.
            next_action_tuple (tuple): The action *actually chosen* in the next state.
        """
        # Convert action tuples to their corresponding single integer indices
        action_index = self._action_tuple_to_index(action_tuple)
        next_action_index = self._action_tuple_to_index(next_action_tuple)

        # Get the current Q-value estimate for the state-action pair
        current_q = self.q_table[state_tuple][action_index]

        # Get the Q-value estimate for the next state and the *next action chosen*
        next_q = self.q_table[next_state_tuple][next_action_index]

        # Calculate the Temporal Difference (TD) target
        td_target = reward + self.gamma * next_q

        # Calculate the TD error
        td_error = td_target - current_q

        # Update the Q-value for the original state-action pair
        new_q = current_q + self.lr * td_error
        self.q_table[state_tuple][action_index] = new_q


    def save_q_table(self, filename="sarsa_q_table.pkl"):
        """
        Saves the learned Q-table to a file using pickle.

        Converts the defaultdict Q-table to a regular dict before saving.

        Args:
            filename (str): The path and name of the file to save the Q-table to.
        """
        # Convert defaultdict to a regular dict for standard pickling
        q_table_dict = dict(self.q_table)
        try:
            with open(filename, 'wb') as f:
                pickle.dump(q_table_dict, f)
            # Use tqdm.write if available (from main script) or print
            try: from tqdm import tqdm; tqdm.write(f"Q-table saved successfully to {filename}")
            except ImportError: print(f"Q-table saved successfully to {filename}")
        except Exception as e:
            try: from tqdm import tqdm; tqdm.write(f"Error saving Q-table to {filename}: {e}")
            except ImportError: print(f"Error saving Q-table to {filename}: {e}")


    def load_q_table(self, filename="sarsa_q_table.pkl"):
        """
        Loads a Q-table from a file using pickle.

        Loads the dictionary and converts it back into the agent's defaultdict Q-table.

        Args:
            filename (str): The path and name of the file to load the Q-table from.

        Returns:
            bool: True if the Q-table was loaded successfully, False otherwise.
        """
        try:
            with open(filename, 'rb') as f:
                q_table_dict = pickle.load(f)

            # Re-initialize the agent's q_table as a defaultdict with the correct structure
            self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
            # Update the defaultdict with the loaded data
            self.q_table.update(q_table_dict)

            print(f"Q-table loaded successfully from {filename}")
            # Verify structure (optional but recommended)
            if q_table_dict:
                 first_key = next(iter(q_table_dict))
                 first_value = q_table_dict[first_key]
                 if not isinstance(first_value, np.ndarray) or len(first_value) != self.num_actions:
                      print(f"Warning: Loaded Q-table structure might be incompatible.")
                      print(f"Expected array length {self.num_actions}, found {len(first_value)} for key {first_key}")
            return True
        except FileNotFoundError:
            print(f"Error: Q-table file '{filename}' not found. Cannot load.")
            return False
        except Exception as e:
            print(f"Error loading Q-table from {filename}: {e}")
            return False
