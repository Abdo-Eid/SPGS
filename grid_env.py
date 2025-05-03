import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import time

# From conf.py
from conf import *
# From utilites.py
from utilites import *


class PowerGridEnv(gym.Env):
    metadata = {'render_modes': ['human']} # Define render metadata

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode # Store render mode

        # --- Environment Timing ---
        self.episode_length_hours = 24
        self.total_timesteps = self.episode_length_hours
        self.step_duration_hours = 1.0 # Assuming 1 hour per step

        # --- System Components ---
        self.main_generators = [
            MainGenerator(f"MainGen{i+1}", MAIN_GEN_MIN_OUTPUT, MAIN_GEN_MAX_OUTPUT, MAIN_GEN_FAIL_PROB_PER_HOUR, MAIN_GEN_HEAL_TIME_HOURS)
            for i in range(N_MAIN_GENS)
        ]
        self.emergency_generators = [
            EmergencyGenerator(f"EmGen{i+1}", EM_GEN_OUTPUT, T_BOOT_EMERGENCY_HOURS, 24 * 7) # Emergency gens have 7 days total runtime
            for i in range(N_EM_GENS)
        ]
        # Pass step_duration_hours to Battery
        self.battery = Battery(
            MAX_BATTERY_CAPACITY, BATTERY_CHARGE_RATE, BATTERY_DISCHARGE_RATE,
            BATTERY_CHARGE_EFFICIENCY, BATTERY_DISCHARGE_EFFICIENCY, self.step_duration_hours
        )

        self.load_zones = {
            'hi': LoadZone("HighPriority", scale=1.0, priority="high"),
            'lo1': LoadZone("LowPriority1", scale=0.7, priority="low1"),
            'lo2': LoadZone("LowPriority2", scale=0.5, priority="low2")
        }
        # Define the order in which load zones are prioritized for power distribution
        self.load_zone_order = ['hi', 'lo1', 'lo2']

        # --- Observation Space ---
        # Structure: [current_time, battery_state, load_states, main_gen_states, em_gen_states]
        # current_time: 1 (normalized)
        # battery_state: 2 (SoC, last_action_mode)
        # load_states: len(zones) * 2 (demand, shed_status for each)
        # main_gen_states: N_MAIN_GENS * 4 (online, fail_timer, min_out, max_out for each)
        # em_gen_states: N_EM_GENS * 3 (online, start_timer, runtime_left for each)
        obs_dim = 1 + 2 + len(self.load_zones) * 2 + N_MAIN_GENS * 4 + N_EM_GENS * 3
        # Using Box(-inf, inf) for simplicity, could use more specific bounds if needed
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Action Space ---
        # Structure: [batt_mode, em1_mode, em2_mode, ..., shed_lo1, shed_lo2]
        # batt_mode: 0=Idle, 1=Discharge, 2=Charge (3 options)
        # em_i_mode: 0=Idle/NoAction, 1=Boot, 2=Shutdown (3 options) for each EM gen
        # shed_lo1: 0=DoNotShed, 1=Shed (2 options)
        # shed_lo2: 0=DoNotShed, 1=Shed (2 options)
        action_space_list = [3] + [3] * N_EM_GENS + [2, 2]
        self.action_space = spaces.MultiDiscrete(action_space_list)

        # --- Internal State Variables for Logging/Rendering ---
        self._total_system_demand = 0.0 # Total demand including battery charge request
        self._total_available_power = 0.0 # Total power from all sources (Gen + Batt Discharge)
        self._power_balance_deficit = 0.0 # Unmet demand + un-charged battery need
        self._main_gen_failures_this_step = np.zeros(N_MAIN_GENS, dtype=bool) # Track failures for reward/info
        self._total_power_consumed_by_loads = 0.0 # Power actually consumed by loads
        self._power_consumed_for_charge = 0.0 # Power actually drawn for battery charge

        # Variable to store the info dictionary from the last step for rendering
        self._last_info = {}


    def _get_obs(self):
        """Constructs the observation array from the current state."""
        # Normalized time (0 to 1)
        obs = [self.current_time / self.episode_length_hours]

        # Battery state
        obs.extend(self.battery.get_state()) # Normalized SoC, Last Mode

        # Load zone states (demand, shed status)
        for zone_key in self.load_zone_order:
             obs.extend(self.load_zones[zone_key].get_state())

        # Main generator states
        for gen in self.main_generators:
             obs.extend(gen.get_state()) # online, fail_timer, min_out, max_out

        # Emergency generator states
        for em_gen in self.emergency_generators:
             obs.extend(em_gen.get_state()) # online, start_timer, runtime_left

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        # --- Reset Time ---
        self.current_time = 0.0

        # --- Reset Components ---
        self.battery.soc = self.battery.max_capacity / 2.0 # Start at half charge
        self.battery.idle() # Reset battery state
        for gen in self.main_generators:
            gen.online = True # Start main gens online
            gen.fail_timer = 0 # No pending failures
        for em_gen in self.emergency_generators:
            em_gen.online = False # Start EM gens offline
            em_gen.start_timer = 0 # Not booting
            em_gen.runtime_left_steps = em_gen.total_runtime_steps # Full runtime available
            em_gen._just_booted_this_step = False # Reset boot flag

        # --- Reset Load Zones ---
        # Update demand for time 0, ensure not shed initially
        for zone in self.load_zones.values():
             zone.update_demand(self.current_time)
             zone.set_shed(False) # Start with no shedding

        # --- Reset Internal Logging/State ---
        self._total_system_demand = 0.0
        self._total_available_power = 0.0
        self._power_balance_deficit = 0.0
        self._main_gen_failures_this_step[:] = False
        self._total_power_consumed_by_loads = 0.0
        self._power_consumed_for_charge = 0.0
        self._last_info = {} # Reset last info

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        """Applies the action and advances the environment by one time step."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # --- Parse Action ---
        a_batt_mode = action[0]
        a_em_modes = action[1 : 1 + N_EM_GENS]
        a_shed_lo1 = action[1 + N_EM_GENS]
        a_shed_lo2 = action[1 + N_EM_GENS + 1]

        # --- Reset Step-Specific State ---
        self._main_gen_failures_this_step[:] = False # Reset main gen failure flags for this step

        # --- Update Component States (pre-power-flow) ---
        # Update Main Generators (handles failures/healing)
        for i, gen in enumerate(self.main_generators):
            if gen.update(): # update() returns True if a failure occurred
                self._main_gen_failures_this_step[i] = True

        # Update Emergency Generators (handles booting/shutdown commands and runtime)
        for i, em_gen in enumerate(self.emergency_generators):
            em_action = a_em_modes[i]
            if em_action == 1: # Command Boot
                 # command_boot sets internal state and returns True if accepted
                 em_gen.command_boot() # Boot cost is handled within command_boot by setting a flag
            elif em_action == 2: # Command Shutdown
                 em_gen.command_shutdown()
            # Update EM gen state (decrements timers, transitions online/offline)
            em_gen.update()

        # Update Load Zones (update demand based on time, apply shedding action)
        for zone_key, zone in self.load_zones.items():
            zone.update_demand(self.current_time)
            if zone_key == 'lo1':
                zone.set_shed(a_shed_lo1)
            elif zone_key == 'lo2':
                zone.set_shed(a_shed_lo2)
            # For HI zone, ensure it's never shed by action, though it might be unmet
            if zone_key == 'hi':
                 zone.set_shed(False) # HI priority cannot be shed by agent action


        # --- Calculate Total System Demand and Available Power ---

        # 1. Calculate total effective demand from loads (after shedding)
        total_effective_load_demand = sum(
            zone.get_effective_demand() for zone in self.load_zones.values()
        )

        # 2. Determine power request for battery charging if actioned
        desired_power_for_battery_charge = 0.0
        if a_batt_mode == 2: # Action to Charge
             desired_power_for_battery_charge = self.battery.charge_rate # Battery will try to draw up to its rate

        # Total demand the system attempts to meet this step
        self._total_system_demand = total_effective_load_demand + desired_power_for_battery_charge # MW

        # 3. Calculate power available from dispatchable sources (EM Gens, Battery Discharge)
        emergency_gen_power = sum(em_gen.get_current_output() for em_gen in self.emergency_generators) # MW

        battery_discharge_power = 0.0
        if a_batt_mode == 1: # Action to Discharge
             # attempt_discharge returns the power actually delivered to the grid
             battery_discharge_power = self.battery.attempt_discharge() # MW
        else:
             # If not discharging, reset battery discharge tracking
             self.battery.idle() # Resets the internal energy tracking for costs

        # 4. Calculate power needed from Main Generators
        # This is the total system demand minus power from EM and Battery Discharge.
        # This value can be negative if EM/Battery discharge provide more than needed.
        required_from_main_gens = self._total_system_demand - emergency_gen_power - battery_discharge_power # MW

        # 5. Calculate actual output from Main Generators
        # Online main gens collectively produce at least their minimum, up to their maximum,
        # trying to meet the 'required_from_main_gens'.
        online_main_gens = [gen for gen in self.main_generators if gen.online]
        num_online_main_gens = len(online_main_gens)
        collective_min_main = num_online_main_gens * MAIN_GEN_MIN_OUTPUT
        collective_max_main = num_online_main_gens * MAIN_GEN_MAX_OUTPUT

        main_gen_actual_output = 0.0
        if num_online_main_gens > 0:
             # Main gens produce at least min, up to max, capped by requirement.
             # If required is less than min, they still produce min (creating surplus).
             main_gen_actual_output = max(collective_min_main, min(required_from_main_gens, collective_max_main))


        # 6. Calculate total available power from all sources
        self._total_available_power = main_gen_actual_output + emergency_gen_power + battery_discharge_power # MW

        # 7. Calculate the theoretical deficit before distribution (Demand vs Available)
        # This is the gap the system cannot fill with its generation + discharge.
        self._power_balance_deficit = max(0.0, self._total_system_demand - self._total_available_power) # MW


        # --- Distribute Available Power to Loads (by Priority) and Battery Charging ---

        power_remaining_for_distribution = self._total_available_power # Start with all available power

        # Distribute to High Priority Loads
        hi_zone = self.load_zones['hi']
        power_consumed_hi = hi_zone.attempt_meet_demand(power_remaining_for_distribution)
        power_remaining_for_distribution -= power_consumed_hi

        # Distribute to Low Priority 1 Loads
        lo1_zone = self.load_zones['lo1']
        power_consumed_lo1 = lo1_zone.attempt_meet_demand(power_remaining_for_distribution)
        power_remaining_for_distribution -= power_consumed_lo1

        # Distribute to Low Priority 2 Loads
        lo2_zone = self.load_zones['lo2']
        power_consumed_lo2 = lo2_zone.attempt_meet_demand(power_remaining_for_distribution)
        power_remaining_for_distribution -= power_consumed_lo2

        # Total power consumed by loads
        self._total_power_consumed_by_loads = power_consumed_hi + power_consumed_lo1 + power_consumed_lo2


        # Distribute remaining power to Battery Charging (if actioned)
        self._power_consumed_for_charge = 0.0 # Power actually drawn for charge this step
        if a_batt_mode == 2: # Action to Charge
             # attempt_charge takes available power and returns what it actually drew
             self._power_consumed_for_charge = self.battery.attempt_charge(available_grid_power=power_remaining_for_distribution)
             power_remaining_for_distribution -= self._power_consumed_for_charge # Update remaining power


        # --- Calculate Reward ---
        reward = 0.0

        # Reward for meeting loads
        if hi_zone.was_met():
            reward += W_HI
        else:
            reward -= C_UNMET_HI # Large penalty for not meeting high priority

        if lo1_zone.was_met():
            reward += W_LO1
        if lo2_zone.was_met():
            reward += W_LO2

        # Cost for shedding loads (applies even if demand was low)
        if lo1_zone.was_shed():
            reward -= C_SHED
        if lo2_zone.was_shed():
            reward -= C_SHED

        # Battery costs (based on energy drawn/delivered)
        energy_drawn_for_charge, energy_discharged = self.battery.get_energy_costs() # MWh
        reward -= C_BATT_DISCHARGE * energy_discharged
        reward -= C_BATT_CHARGE * energy_drawn_for_charge

        # Emergency generator costs
        for i, em_gen in enumerate(self.emergency_generators):
            # Cost for booting (applied only in the step the command_boot is accepted)
            if em_gen._just_booted_this_step:
                 reward -= C_EM_BOOT

            # Cost for running (applied every step it is online)
            if em_gen.online:
                 reward -= C_EM_RUN * self.step_duration_hours # Cost per hour running

                 # Cost for being online but effectively idle (system had surplus power after meeting needs)
                 # Check if the total power consumed (loads + battery charge) was less than the total available power
                 total_power_used = self._total_power_consumed_by_loads + self._power_consumed_for_charge
                 # Use a small tolerance for floating point comparisons
                 if em_gen.online and self._total_available_power > total_power_used + 1e-6:
                      reward -= C_EM_IDLE_ONLINE * self.step_duration_hours # Cost per hour idle online


        # Main generator failure cost (penalty per failure incident)
        reward -= np.sum(self._main_gen_failures_this_step) * C_FAIL


        # --- Advance Time ---
        self.current_time += self.step_duration_hours

        # --- Check Termination Conditions ---
        # Episode ends after a fixed duration
        terminated = self.current_time >= self.episode_length_hours
        truncated = False # No truncation condition specified yet

        # --- Construct Info Dictionary ---
        info = {
            'current_time': self.current_time,
            'total_demand_MW': self._total_system_demand, # Total demand including battery charge request
            'available_power_MW': self._total_available_power, # Total power available from sources
            'power_balance_deficit_MW': self._power_balance_deficit, # Deficit = TotalDemand - TotalAvailable
            'power_consumed_loads_MW': self._total_power_consumed_by_loads,
            'power_consumed_charge_MW': self._power_consumed_for_charge,
            'loads_demand_MW': {zone_key: zone._current_demand for zone_key, zone in self.load_zones.items()}, # Actual demand
            'loads_effective_demand_MW': {zone_key: zone.get_effective_demand() for zone_key, zone in self.load_zones.items()}, # Demand after shedding
            'loads_met': {zone_key: zone.was_met() for zone_key, zone in self.load_zones.items()},
            'loads_shed': {'lo1': lo1_zone.was_shed(), 'lo2': lo2_zone.was_shed()},
            'battery_soc_MWh': self.battery.soc,
            'battery_action_mode': a_batt_mode, # Action mode from agent
            'energy_drawn_for_charge_MWh': energy_drawn_for_charge, # Energy drawn for cost
            'energy_discharged_MWh': energy_discharged, # Energy discharged for cost/benefit
            'main_gen_online': [gen.online for gen in self.main_generators],
            'main_gen_failures_this_step': self._main_gen_failures_this_step.tolist(),
            'emergency_gens_online': [em_gen.online for em_gen in self.emergency_generators],
            'emergency_gens_booting': [em_gen.start_timer > 0 for em_gen in self.emergency_generators],
            'emergency_gens_runtime_left': [em_gen.runtime_left_steps for em_gen in self.emergency_generators],
        }

        # Add critical failure flag to info if HI load was not met
        if not hi_zone.was_met():
            info['critical_failure'] = True

        # Store the info dictionary for the next render call
        self._last_info = info


        # --- Construct Observation ---
        observation = self._get_obs()

        # Render if render_mode is set
        if self.render_mode is not None:
             self.render()


        return observation, reward, terminated, truncated, info

    def render(self, RENDER_SLEEP_TIME = 5):
        """Renders the current state of the environment using the last info dictionary."""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying render_mode "
                "in the PowerGridEnv constructor. Reinforcement Learning "
                "environments in Gymnasium typically support multiple render modes, "
                "most commonly 'human' for interactive display and 'rgb_array' "
                "for training integration. Pass an appropriate render_mode option "
                "to the environment constructor."
            )
            return

        # Use the stored info dictionary
        info = self._last_info
        if not info:
             # Handle case where render is called before the first step
             print("Render called before the first step. No info available yet.")
             return

        # Basic command line rendering
        os.system("cls" if os.name == "nt" else "clear")
        print("⚡ Power Grid Dashboard")
        print(f"Time: {info.get('current_time', self.current_time):.2f} / {self.episode_length_hours:.2f} hours | Step Duration: {self.step_duration_hours:.2f} hours")
        print("=" * 60)

        print("\n📊 Power Balance:")
        print(f"  Total System Demand: {info.get('total_demand_MW', 0.0):.2f} MW")
        print(f"  Total Available Power: {info.get('available_power_MW', 0.0):.2f} MW")
        print(f"  Power Deficit: {info.get('power_balance_deficit_MW', 0.0):.2f} MW")
        print("-" * 60)

        print("\n🏘️ Load Zones:")
        # Use info dict to get states accurately reflecting the last step
        loads_demand = info.get('loads_demand_MW', {})
        loads_effective_demand = info.get('loads_effective_demand_MW', {})
        loads_met = info.get('loads_met', {})
        loads_shed = info.get('loads_shed', {})

        for zone_key in self.load_zone_order:
            zone = self.load_zones[zone_key] # Get zone object for name/priority
            met_status = loads_met.get(zone_key, False)
            shed_status = loads_shed.get(zone_key, False) if zone_key in ['lo1', 'lo2'] else False # HI is never shed by action

            status = "MET" if met_status else ("SHED" if shed_status else "UNMET")
            print(f"  {zone.name} ({zone.priority}): Demand={loads_demand.get(zone_key, 0.0):.2f} MW | Effective={loads_effective_demand.get(zone_key, 0.0):.2f} MW | Status: {status}")
        print("-" * 60)

        print("\n🔋 Battery:")
        mode_map = {0: "Idle", 1: "Discharging", 2: "Charging"}
        print(f"  SoC: {info.get('battery_soc_MWh', self.battery.soc):.2f}/{self.battery.max_capacity:.2f} MWh ({info.get('battery_soc_MWh', self.battery.soc)/self.battery.max_capacity:.1%})")
        print(f"  Mode (Action This Step): {mode_map.get(info.get('battery_action_mode', -1), 'Unknown')}") # Use action from info
        print(f"  Energy Drawn (Charge Cost): {info.get('energy_drawn_for_charge_MWh', 0.0):.2f} MWh | Energy Discharged (Benefit/Cost): {info.get('energy_discharged_MWh', 0.0):.2f} MWh")
        print("-" * 60)

        print("\n⚡ Generators:")
        main_gen_online = info.get('main_gen_online', [gen.online for gen in self.main_generators])
        main_gen_failures_this_step = info.get('main_gen_failures_this_step', [False] * N_MAIN_GENS)
        online_main_count = sum(main_gen_online)
        print(f"  --- Main --- Online: {online_main_count}/{N_MAIN_GENS}")
        for i, gen in enumerate(self.main_generators):
            status = "Online" if main_gen_online[i] else f"Offline (Heal: {gen.fail_timer}/{gen.heal_time_steps} steps)"
            failure_msg = " !!! Failed THIS Step !!!" if main_gen_failures_this_step[i] else ""
            print(f"  {gen.name}: Status={status}{failure_msg}")

        emergency_gens_online = info.get('emergency_gens_online', [em_gen.online for em_gen in self.emergency_generators])
        emergency_gens_booting = info.get('emergency_gens_booting', [em_gen.start_timer > 0 for em_gen in self.emergency_generators])
        emergency_gens_runtime_left = info.get('emergency_gens_runtime_left', [em_gen.runtime_left_steps for em_gen in self.emergency_generators])
        online_em_count = sum(emergency_gens_online)
        print(f"  --- Emergency --- Online: {online_em_count}/{N_EM_GENS}")
        for i, em_gen in enumerate(self.emergency_generators):
            status = "Online" if emergency_gens_online[i] else (f"Booting ({em_gen.start_timer}/{em_gen.boot_time_steps} steps left)" if emergency_gens_booting[i] else "Offline")
            print(f"  {em_gen.name}: Status={status} | Output: {em_gen.output_power if emergency_gens_online[i] else 0.0:.2f} MW | Runtime Left: {emergency_gens_runtime_left[i]} steps")
        print("=" * 60)

        # Add sleep after clearing the screen
        time.sleep(RENDER_SLEEP_TIME)
        
    def close(self):
        """Cleans up resources (currently none needed)."""
        pass