# --- Environment Constants ---
N_MAIN_GENS = 3 # Example number of main generators
N_EM_GENS = N_MAIN_GENS # Assume one emergency gen per main for now
MAX_BATTERY_CAPACITY = 100 # Example MWh
BATTERY_CHARGE_RATE = 10 # Example MW
BATTERY_DISCHARGE_RATE = 15 # Example MW (can be different)
BATTERY_CHARGE_EFFICIENCY = 0.9 # Example
BATTERY_DISCHARGE_EFFICIENCY = 0.9 # Example

T_BOOT_EMERGENCY_HOURS = 2 # Example hours (steps) to boot
EM_GEN_OUTPUT = 20 # Example MW fixed output per emergency gen
MAIN_GEN_MIN_OUTPUT = 10 # Example MW
MAIN_GEN_MAX_OUTPUT = 50 # Example MW
MAIN_GEN_FAIL_PROB_PER_HOUR = 0.06 # Example probability of failure per hour
MAIN_GEN_HEAL_TIME_HOURS = 3 # Example hours (steps) to heal

# Reward weights and costs
W_HI = 10.0
W_LO1 = 5.0
W_LO2 = 3.0
C_BATT_DISCHARGE = 0.01 # Cost per MWh discharged (reflects efficiency or wear)
C_BATT_CHARGE = 0.02 # Cost per MWh charged (reflects adding load) - Cost is per MWh DRAWN from grid
C_EM_BOOT = 5.0
C_EM_RUN = 0.5 # Cost per hour running
C_EM_IDLE_ONLINE = 0.1 # Cost per hour online but not used (optional, can be zero)
C_FAIL = 20.0
C_SHED = 15.0
C_UNMET_HI = 50.0 # Large penalty for not meeting high priority demand

# Render settings
# RENDER_SLEEP_TIME = 5 # Seconds to pause during human rendering
