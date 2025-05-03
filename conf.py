# --- Generator Settings ---
N_MAIN_GENS = 3         # Number of main generators in the grid.
N_EM_GENS = N_MAIN_GENS # Number of emergency generators (assumed one per main generator).

MAIN_GEN_MIN_OUTPUT = 10 # Minimum power output (MW) when a main generator is online.
MAIN_GEN_MAX_OUTPUT = 50 # Maximum power output (MW) a single main generator can provide.
MAIN_GEN_FAIL_PROB_PER_HOUR = 0.03 # Probability that an online main generator fails during a 1-hour step.
MAIN_GEN_HEAL_TIME_HOURS = 3 # Number of hours (steps) a main generator stays offline to heal after failing.

EM_GEN_OUTPUT = 20       # Fixed power output (MW) of a single emergency generator when online.
T_BOOT_EMERGENCY_HOURS = 2 # Number of hours (steps) required for an emergency generator to become online after being commanded to boot.
# Note: Emergency generators also have a limited total runtime defined in grid_env.py initialization.

# --- Battery Settings ---
MAX_BATTERY_CAPACITY = 100 # Maximum energy storage capacity of the battery (MWh).
BATTERY_CHARGE_RATE = 10   # Maximum power (MW) the battery can draw from the grid when charging.
BATTERY_DISCHARGE_RATE = 15 # Maximum power (MW) the battery can deliver to the grid when discharging.
BATTERY_CHARGE_EFFICIENCY = 0.9 # Efficiency factor for charging (Energy Stored / Energy Drawn). E.g., 0.9 means 10 MWh drawn stores 9 MWh.
BATTERY_DISCHARGE_EFFICIENCY = 0.9 # Efficiency factor for discharging (Energy Delivered / Energy Removed from SoC). E.g., 0.9 means removing 10 MWh from SoC delivers 9 MWh to the grid.

# --- Reward Function Weights and Costs ---
# These values determine the agent's objectives. Higher weights encourage meeting that demand,
# higher costs discourage that action/event.

# Positive Rewards (for meeting demand)
W_HI = 10.0   # Reward multiplier for successfully meeting high-priority load demand each step.
W_LO1 = 5.0   # Reward multiplier for successfully meeting low-priority zone 1 demand each step.
W_LO2 = 3.0   # Reward multiplier for successfully meeting low-priority zone 2 demand each step.

# Negative Rewards / Costs (Penalties)
C_BATT_DISCHARGE = 0.01 # Cost per MWh discharged from the battery (e.g., reflects wear or opportunity cost).
C_BATT_CHARGE = 0.02    # Cost per MWh drawn *from the grid* to charge the battery (reflects adding load or energy loss).
C_EM_BOOT = 5.0         # One-time cost incurred when an emergency generator starts its boot sequence.
C_EM_RUN = 0.5          # Cost per hour for each emergency generator that is online and running.
C_EM_IDLE_ONLINE = 0.1  # Cost per hour for each emergency generator that is online but not strictly needed (i.e., system had surplus power).
C_FAIL = 20.0           # Penalty incurred each time a main generator fails.
C_SHED = 15.0           # Penalty incurred for each low-priority zone that is actively shed by the agent's action.
C_UNMET_HI = 50.0       # Large penalty incurred if the high-priority load demand is not fully met.

# --- Render settings ---
# RENDER_SLEEP_TIME = 5 # Seconds to pause during human rendering (Now handled within render method in grid_env.py)
# Note: The actual sleep time is now an argument to the render() method itself.

