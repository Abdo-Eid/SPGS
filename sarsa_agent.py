import numpy as np
import pickle
from collections import defaultdict
import gymnasium as gym # Need for action space type check

class SARSAgent:
    """
    A simplified SARSA agent (constant epsilon) for environments with
    discrete actions and potentially continuous states (requiring discretization).
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.8,
                 epsilon=0.1, num_bins_per_feature=None):
        """
        Initializes the SARSA Agent with a constant exploration rate.

        Args:
            env: The environment instance.
            learning_rate (float): Alpha parameter.
            discount_factor (float): Gamma parameter.
            epsilon (float): Constant exploration rate (0 <= epsilon <= 1).
            num_bins_per_feature (tuple/list): A tuple defining the number of bins
                                              for each feature in the observation space
                                              that needs discretization. Must match the
                                              structure decided for discretization.
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon # Constant epsilon

        # Action space details
        if not isinstance(env.action_space, gym.spaces.MultiDiscrete):
            raise ValueError("SARSAgent currently requires MultiDiscrete action space.")
        self.action_nvec = env.action_space.nvec
        self.num_actions = np.prod(self.action_nvec).item() # Total number of unique action combinations

        # State space discretization details
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        # --- IMPORTANT: Define discretization bins ---
        # This needs careful alignment with the structure of _get_obs()
        # Example structure (adjust based on actual obs structure and needs):
        if num_bins_per_feature is None:
            # Define a default binning strategy - THIS IS A GUESS and likely needs tuning!
            # [time, batt_soc, batt_mode, loads..., main_gens..., em_gens...]
            bins = [10] # time (0-1)
            bins += [10, 3] # batt_soc (0-1), batt_mode (0,1,2)
            # Loads (hi_demand, hi_shed(ignored), lo1_demand, lo1_shed, lo2_demand, lo2_shed)
            # Estimate demand ranges (can be refined by observing env runs)
            # Max demand from profile ~ 110 (scale 1), ~77 (scale 0.7), ~55 (scale 0.5)
            bins += [10, 1, 10, 2, 10, 2] # Ignoring hi_shed (1 bin), using 2 for lo_shed
            self.load_demand_bins = [(0, 120), (0, 80), (0, 60)] # Approx ranges for hi, lo1, lo2
            # Main Gens (online, fail_timer, min_out(ignored), max_out(ignored)) * N
            main_heal_bins = env.main_generators[0].heal_time_steps + 1
            for _ in range(env.main_generators.__len__()):
                 bins += [2, main_heal_bins, 1, 1] # Ignoring min/max output (1 bin each)
            # Em Gens (online, start_timer, runtime_left) * N
            em_boot_bins = env.emergency_generators[0].boot_time_steps + 1
            em_runtime_bins = 10 # Bin runtime into 10 levels
            self.em_runtime_max = env.emergency_generators[0].total_runtime_steps
            for _ in range(env.emergency_generators.__len__()):
                bins += [2, em_boot_bins, em_runtime_bins]
            self.num_bins = tuple(bins)
            print(f"Using default discretization bins: {self.num_bins}")
            print(f"Estimated number of states: {np.prod(self.num_bins)}")
        else:
            self.num_bins = tuple(num_bins_per_feature)
            # Note: If using custom bins, ensure load/runtime ranges are handled if needed

        if len(self.num_bins) != env.observation_space.shape[0]:
             raise ValueError(f"Length of num_bins ({len(self.num_bins)}) must match observation space dimension ({env.observation_space.shape[0]})")

        # Use defaultdict for Q-table to handle unseen states gracefully
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        # Precompute factors for action index mapping
        self._action_map_factors = np.cumprod(self.action_nvec[::-1])[::-1] // self.action_nvec

    def _action_tuple_to_index(self, action_tuple):
        """Maps a MultiDiscrete action tuple to a single integer index."""
        if len(action_tuple) != len(self.action_nvec):
            raise ValueError("Action tuple length mismatch.")
        index = 0
        for i, a in enumerate(action_tuple):
             if not (0 <= a < self.action_nvec[i]):
                  raise ValueError(f"Action element {a} at index {i} is out of bounds {self.action_nvec[i]}.")
             factor = 1
             if i < len(action_tuple) - 1:
                  factor = np.prod(self.action_nvec[i+1:]).item()
             index += a * factor
        return int(index)


    def _action_index_to_tuple(self, index):
        """Maps a single integer index back to a MultiDiscrete action tuple."""
        if not (0 <= index < self.num_actions):
            raise ValueError(f"Action index {index} out of bounds {self.num_actions}.")
        action_tuple = [0] * len(self.action_nvec)
        remainder = index
        for i in range(len(self.action_nvec)):
            factor = 1
            if i < len(self.action_nvec) - 1:
                 factor = np.prod(self.action_nvec[i+1:]).item()
            action_val = remainder // factor
            action_tuple[i] = action_val
            remainder %= factor
        return tuple(action_tuple)

    def discretize_state(self, observation):
        """Discretizes a continuous observation into a tuple of bin indices."""
        state = [0] * len(self.num_bins)
        obs_idx = 0 # Keep track of index in the raw observation vector

        # 1. Time (1 feature)
        state[obs_idx] = np.digitize(np.clip(observation[obs_idx], 0, 1), np.linspace(0, 1, self.num_bins[obs_idx], endpoint=False)) -1
        obs_idx += 1

        # 2. Battery (2 features: soc, mode)
        state[obs_idx] = np.digitize(np.clip(observation[obs_idx], 0, 1), np.linspace(0, 1, self.num_bins[obs_idx], endpoint=False)) - 1 # soc
        obs_idx += 1
        state[obs_idx] = int(observation[obs_idx]) # mode (already discrete 0, 1, 2)
        obs_idx += 1

        # 3. Load Zones (2 features per zone: demand, shed)
        load_zone_keys = ['hi', 'lo1', 'lo2']
        for i, zone_key in enumerate(load_zone_keys):
             # Demand
             demand_val = observation[obs_idx]
             min_demand, max_demand = self.load_demand_bins[i]
             bins = np.linspace(min_demand, max_demand, self.num_bins[obs_idx], endpoint=False)
             state[obs_idx] = np.digitize(np.clip(demand_val, min_demand, max_demand), bins) - 1
             obs_idx += 1
             # Shed
             state[obs_idx] = int(observation[obs_idx]) # shed (already discrete 0, 1)
             obs_idx += 1

        # 4. Main Generators (4 features per gen: online, fail_timer, min_out, max_out)
        for i in range(len(self.env.main_generators)):
             state[obs_idx] = int(observation[obs_idx]) # online
             obs_idx += 1
             # fail_timer (integer 0 to heal_time)
             timer_val = int(observation[obs_idx])
             max_timer = self.env.main_generators[i].heal_time_steps
             state[obs_idx] = np.clip(timer_val, 0, max_timer) # Use directly as bin index
             obs_idx += 1
             state[obs_idx] = 0 # min_out (ignored, 1 bin)
             obs_idx += 1
             state[obs_idx] = 0 # max_out (ignored, 1 bin)
             obs_idx += 1

        # 5. Emergency Generators (3 features per gen: online, start_timer, runtime_left)
        for i in range(len(self.env.emergency_generators)):
            state[obs_idx] = int(observation[obs_idx]) # online
            obs_idx += 1
            # start_timer (integer 0 to boot_time)
            timer_val = int(observation[obs_idx])
            max_timer = self.env.emergency_generators[i].boot_time_steps
            state[obs_idx] = np.clip(timer_val, 0, max_timer) # Use directly as bin index
            obs_idx += 1
            # runtime_left (needs binning)
            runtime_val = observation[obs_idx]
            bins = np.linspace(0, self.em_runtime_max, self.num_bins[obs_idx], endpoint=False)
            state[obs_idx] = np.digitize(np.clip(runtime_val, 0, self.em_runtime_max), bins) - 1
            obs_idx += 1

        # Ensure all bins are non-negative after subtraction
        state = [max(0, s) for s in state]

        return tuple(state)


    def choose_action(self, state_tuple):
        """Chooses an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action index
            action_index = np.random.randint(self.num_actions)
        else:
            # Exploit: choose the best action index from Q-table
            q_values = self.q_table[state_tuple]
            # Handle ties by choosing randomly among the best actions
            best_action_indices = np.where(q_values == np.max(q_values))[0]
            action_index = np.random.choice(best_action_indices)

        # Convert the chosen index back to the action tuple needed by the environment
        action_tuple = self._action_index_to_tuple(action_index)
        return action_tuple

    def update(self, state_tuple, action_tuple, reward, next_state_tuple, next_action_tuple):
        """Updates the Q-table using the SARSA update rule."""
        action_index = self._action_tuple_to_index(action_tuple)
        next_action_index = self._action_tuple_to_index(next_action_tuple)

        current_q = self.q_table[state_tuple][action_index]
        next_q = self.q_table[next_state_tuple][next_action_index]

        # SARSA update formula
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.q_table[state_tuple][action_index] = new_q


    def save_q_table(self, filename="sarsa_q_table.pkl"):
        """Saves the Q-table to a file."""
        # Convert defaultdict to a regular dict for saving
        q_table_dict = dict(self.q_table)
        try:
            with open(filename, 'wb') as f:
                pickle.dump(q_table_dict, f)
            print(f"Q-table saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load_q_table(self, filename="sarsa_q_table.pkl"):
        """Loads the Q-table from a file."""
        try:
            with open(filename, 'rb') as f:
                q_table_dict = pickle.load(f)
            # Convert back to defaultdict
            self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
            self.q_table.update(q_table_dict)
            print(f"Q-table loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Error: Q-table file '{filename}' not found.")
            return False
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return False

