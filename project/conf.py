# --- Constants ---
# Grid component types
GENERATOR = 'generator'
SUBSTATION = 'substation'
LOAD_ZONE = 'load_zone'

# Connection status
IN_SERVICE = 'in_service'
OUT_OF_SERVICE = 'out_of_service'

# Simulation Parameters
MAX_STEPS_PER_EPISODE = 100
PROB_LINE_FAILURE = 0.03  # Probability a line fails each step
PROB_LINE_REPAIR = 0.10   # Probability an out-of-service line is repaired

# RL Parameters (SARSA)
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.7   # Discount factor for future rewards
EXPLORATION_RATE = 0.1  # Epsilon (constant exploration)

REWARD_LOAD_POWERED = 1.0
REWARD_CRITICAL_LOAD_POWERED = 5.0
PENALTY_LOAD_BLACKOUT = -2.0
PENALTY_CRITICAL_LOAD_BLACKOUT = -10.0
PENALTY_SWITCHING_ACTION = -0.1

MODEL_FILENAME = "power_grid_sarsa_qtable.npy" # Using .npy for numpy save

# Visualization Parameters
VISUALIZATION_PAUSE = 0.2 # Seconds between steps in demo mode

