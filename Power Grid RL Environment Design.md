# Power Grid RL Environment Design (Reflecting Implementation)

This document outlines the design of the Power Grid Reinforcement Learning environment, updated to match the implementation in `grid_env.py`, `utilites.py`, and `conf.py`.

### **1. Environment Components**

*   **N main generators (`N_MAIN_GENS`):** Each has min/max output, failure probability, and heal time. Output is determined by grid need between min/max when online.
*   **N emergency generators (`N_EM_GENS`):** Agent can command boot (takes `T_BOOT_EMERGENCY_HOURS`) or shutdown. Fixed output (`EM_GEN_OUTPUT`) when online. Limited total runtime.
*   **1 battery:** Agent controls mode (Idle, Charge, Discharge). Has capacity, charge/discharge rates, and efficiencies. Charging adds to system demand.
*   **1 high-priority load zone ('hi'):** Cannot be shed by agent action. Must be met if possible.
*   **2 low-priority load zones ('lo1', 'lo2'):** Agent can choose to shed these zones (`a_shed_lo1`, `a_shed_lo2`). Different demand scaling factors.
*   **Dynamic Demand:** Based on a daily profile (`demand_profile` function), scaled per zone.
*   **Simulation:** Runs in discrete time steps (e.g., 1 hour per step) for a fixed episode duration (e.g., 24 hours).

---

### **2. Load Zone Demand Functions**

Each zone's demand `D` at time `t` (hours) is based on `demand_profile(t, scale) * 100` (MW).

```python
# From utilites.py
def demand_profile(t, scale=1.0):
    t_wrapped = t % 24
    base = 0.5
    morning_peak = 0.4 * np.exp(-(t_wrapped - 8)**2 / (2 * 2**2))
    evening_peak = 0.6 * np.exp(-(t_wrapped - 19)**2 / (2 * 2**2))
    return scale * (base + morning_peak + evening_peak) * 100 # Scale factor applied

# Example usage in environment:
D_hi = demand_profile(current_time, scale=1.0) * 100
D_lo1 = demand_profile(current_time, scale=0.7) * 100
D_lo2 = demand_profile(current_time, scale=0.5) * 100
```

---

### **3. Generator Behavior**

*   **Main Generators:**
    *   Output determined by `np.clip(required_power, collective_min, collective_max)` when online.
    *   Fail with probability `MAIN_GEN_FAIL_PROB_PER_HOUR` per step.
    *   Offline for `MAIN_GEN_HEAL_TIME_HOURS` steps after failure.
*   **Emergency Generators:**
    *   Agent action `a_em[i]`: 0 (Idle), 1 (Boot), 2 (Shutdown).
    *   Boot takes `T_BOOT_EMERGENCY_HOURS` steps (timer decrements).
    *   Fixed output `EM_GEN_OUTPUT` when online.
    *   `runtime_left_steps` decrements each step while online. Auto-shutdown when runtime reaches 0.

---

### **4. Battery Dynamics**

*   SoC (State of Charge) ∈ [0, `MAX_BATTERY_CAPACITY`] (MWh).
*   Agent action `a_batt`: 0 (Idle), 1 (Discharge), 2 (Charge).
*   **Discharge:** Delivers power up to `BATTERY_DISCHARGE_RATE` (MW), limited by SoC and `BATTERY_DISCHARGE_EFFICIENCY`. Decreases SoC.
*   **Charge:** Draws power up to `BATTERY_CHARGE_RATE` (MW), limited by available grid power, remaining capacity, and `BATTERY_CHARGE_EFFICIENCY`. Increases SoC. Charging power *adds* to total system demand.

---

### **5. Observation Space (Implemented)**

The observation is a flat `np.ndarray` (dtype `float32`) returned by `env._get_obs()`. Its structure is crucial for the agent's `discretize_state` method.

| Index Range           | Description                         | Feature Count | Notes                                      |
| :-------------------- | :---------------------------------- | :------------ | :----------------------------------------- |
| `[0]`                 | Time (normalized)                   | 1             | `current_time / episode_length_hours`      |
| `[1]`                 | Battery SoC (normalized)            | 1             | `soc / max_capacity`                       |
| `[2]`                 | Battery Last Action Mode            | 1             | 0.0 (Idle), 1.0 (Discharge), 2.0 (Charge)  |
| `[3]`                 | Load Zone 'hi' Demand (MW)          | 1             | `_current_demand`                          |
| `[4]`                 | Load Zone 'hi' Shed Status          | 1             | Always 0.0                                 |
| `[5]`                 | Load Zone 'lo1' Demand (MW)         | 1             | `_current_demand`                          |
| `[6]`                 | Load Zone 'lo1' Shed Status         | 1             | 0.0 (Not Shed), 1.0 (Shed)                 |
| `[7]`                 | Load Zone 'lo2' Demand (MW)         | 1             | `_current_demand`                          |
| `[8]`                 | Load Zone 'lo2' Shed Status         | 1             | 0.0 (Not Shed), 1.0 (Shed)                 |
| `[9:9+4*N_MAIN]`      | Main Generators (N_MAIN_GENS * 4)   | `4*N_MAIN`    | Per Gen: [online, fail_timer, min_out, max_out] |
| `[9+4*N_MAIN:end]`    | Emergency Generators (N_EM_GENS * 3)| `3*N_EM`      | Per Gen: [online, start_timer, runtime_left] |

**Total Observation Dimension:** `1 + 2 + (2 * num_zones) + (4 * N_MAIN_GENS) + (3 * N_EM_GENS)`
(Assuming 3 zones: `1 + 2 + 6 + 4*N_MAIN + 3*N_EM = 9 + 4*N_MAIN + 3*N_EM`)

---

### **6. Action Space (Implemented)**

The action space is `gym.spaces.MultiDiscrete`. The agent provides an action as a tuple or list.

| Index | Component         | Options | Values        | Notes                           |
| :---- | :---------------- | :------ | :------------ | :------------------------------ |
| `[0]` | Battery Mode      | 3       | `0, 1, 2`     | Idle, Discharge, Charge         |
| `[1]` | EM Gen 1 Mode     | 3       | `0, 1, 2`     | Idle/NoOp, Boot, Shutdown       |
| `...` | ...               | ...     | ...           | ...                             |
| `[N_EM]`| EM Gen N_EM Mode  | 3       | `0, 1, 2`     | Idle/NoOp, Boot, Shutdown       |
| `[N_EM+1]`| Shed Load Zone 1| 2       | `0, 1`        | No Shed, Shed                   |
| `[N_EM+2]`| Shed Load Zone 2| 2       | `0, 1`        | No Shed, Shed                   |

**Action Vector Length:** `1 + N_EM_GENS + 2`

---

### **7. Power Allocation Logic (Implemented)**

Executed within `env.step()`:

1.  **Calculate Effective Demand:** Sum demand of non-shed loads (`D_hi + D_lo1*(not shed) + D_lo2*(not shed)`).
2.  **Calculate Charge Request:** If `a_batt == 2`, add `BATTERY_CHARGE_RATE` to potential demand.
3.  **Calculate Total System Demand:** Effective Load Demand + Charge Request.
4.  **Calculate Available Dispatchable Power:** Sum output of online EM Gens + Battery Discharge (if `a_batt == 1`, considering rate, SoC, efficiency).
5.  **Calculate Required Main Gen Power:** Total System Demand - Dispatchable Power.
6.  **Calculate Actual Main Gen Power:** Clip required power within the collective `[min_output, max_output]` of *online* main generators. Ensure output is at least collective `min_output`.
7.  **Calculate Total Available Power:** Actual Main Gen Power + Dispatchable Power.
8.  **Distribute Power (Priority Order):**
    *   Meet 'hi' zone demand.
    *   Meet 'lo1' zone demand (if not shed).
    *   Meet 'lo2' zone demand (if not shed).
    *   Use remaining power for Battery Charge (if `a_batt == 2`, limited by rate, capacity, efficiency).
9.  **Track Consumption & Deficit:** Record power consumed by loads, power used for charging, and the overall deficit (`max(0, Total System Demand - Total Available Power)`).

---

### **8. Reward Function (Implemented)**

Calculated at the end of each `env.step()` based on `conf.py` constants:

```python
# Conceptual breakdown (see grid_env.py for exact calculation)
reward = 0.0

# Rewards for meeting demand (check zone.was_met() after power distribution)
if hi_zone.was_met(): reward += W_HI
if lo1_zone.was_met(): reward += W_LO1
if lo2_zone.was_met(): reward += W_LO2

# Penalties for unmet demand / shedding actions
if not hi_zone.was_met() and hi_zone.get_effective_demand() > 1e-6: reward -= C_UNMET_HI # Critical failure
if lo1_zone.was_shed(): reward -= C_SHED # Penalty for shedding action
if lo2_zone.was_shed(): reward -= C_SHED # Penalty for shedding action

# Costs for battery usage (based on energy transferred in the step)
energy_drawn_MWh, energy_discharged_MWh = battery.get_energy_costs()
reward -= C_BATT_CHARGE * energy_drawn_MWh
reward -= C_BATT_DISCHARGE * energy_discharged_MWh

# Costs/Penalties for generators
for em_gen in emergency_generators:
    if em_gen._just_booted_this_step: reward -= C_EM_BOOT # Boot cost
    if em_gen.online:
        reward -= C_EM_RUN * step_duration_hours # Running cost
        # Idle cost if online but system had surplus power
        if total_available_power > (total_power_consumed_by_loads + power_consumed_for_charge + 1e-6):
             reward -= C_EM_IDLE_ONLINE * step_duration_hours

# Penalty for main generator failures
reward -= C_FAIL * sum(main_gen_failures_this_step)
```

Key penalties/costs include: not meeting high-priority demand (`C_UNMET_HI`), shedding low-priority loads (`C_SHED`), battery charge/discharge (`C_BATT_...`), emergency generator booting/running/idling (`C_EM_...`), and main generator failures (`C_FAIL`). Positive rewards incentivize meeting demand (`W_HI`, `W_LO1`, `W_LO2`).