"""
Power Grid Simulation Environment using Gymnasium API.

This module defines the `PowerGridEnv` class, which simulates a simplified
power grid with main generators, emergency generators, battery storage, and
multiple load zones with varying priorities. The environment follows the
Gymnasium interface, allowing it to be used with standard RL algorithms.

The agent's goal is to manage the grid (battery actions, emergency generator
deployment, load shedding) to meet demand, minimize costs, and avoid penalties
over a simulated period (e.g., 24 hours).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import time

# Import constants from conf.py
from conf import *
# Import utility classes (Generators, Battery, LoadZone) from utilites.py
from utilites import *


class PowerGridEnv(gym.Env):
    """
    A Gymnasium environment simulating a power grid with generation, storage, and loads.

    Inherits from `gym.Env` and implements the standard API methods:
    `step`, `reset`, `render`, `close`.

    Attributes:
        render_mode (str | None): Mode for rendering ('human', 'terminal', or None).
        main_generators (list[MainGenerator]): List of main generator objects.
        emergency_generators (list[EmergencyGenerator]): List of emergency generator objects.
        battery (Battery): Battery storage object.
        load_zones (dict[str, LoadZone]): Dictionary mapping zone keys ('hi', 'lo1', 'lo2') to LoadZone objects.
        load_zone_order (list[str]): Order of priority for load distribution.
        observation_space (spaces.Box): Defines the structure and bounds of observations.
        action_space (spaces.MultiDiscrete): Defines the structure of agent actions.
        current_time (float): Current time within the episode (hours).
        episode_length_hours (int): Duration of one episode in hours.
        total_timesteps (int): Total number of steps per episode.
        step_duration_hours (float): Duration of a single time step in hours.
        _last_info (dict): Stores the info dictionary from the last step for rendering.
        # Internal state variables for tracking metrics within a step:
        _total_system_demand (float): Total demand including loads and battery charge request (MW).
        _total_available_power (float): Total power from all online sources (MW).
        _power_balance_deficit (float): Unmet demand (MW).
        _main_gen_failures_this_step (np.ndarray): Boolean array tracking main gen failures this step.
        _total_power_consumed_by_loads (float): Power actually consumed by loads (MW).
        _power_consumed_for_charge (float): Power actually drawn for battery charging (MW).
    """
    metadata = {'render_modes': ['human', 'terminal']} # Define supported render modes

    def __init__(self, render_mode=None, RENDER_SLEEP_TIME = 2.5):
        """
        Initializes the Power Grid Environment.

        Args:
            render_mode (str | None): Specifies the rendering mode.
                'human': Displays a formatted dashboard (clears screen).
                'terminal': Prints step summaries sequentially.
                None: No rendering.
                Defaults to None.
        """
        super().__init__()
        self.render_mode = render_mode # Store render mode
        self.RENDER_SLEEP_TIME = RENDER_SLEEP_TIME

        # --- Environment Timing ---
        self.episode_length_hours = 24 # Duration of one simulation episode
        self.total_timesteps = self.episode_length_hours # Steps per episode
        self.step_duration_hours = 1.0 # Duration of each time step

        # --- System Components Initialization ---
        self.main_generators = [
            MainGenerator(f"MainGen{i+1}", MAIN_GEN_MIN_OUTPUT, MAIN_GEN_MAX_OUTPUT, MAIN_GEN_FAIL_PROB_PER_HOUR, MAIN_GEN_HEAL_TIME_HOURS)
            for i in range(N_MAIN_GENS)
        ]
        self.emergency_generators = [
            EmergencyGenerator(f"EmGen{i+1}", EM_GEN_OUTPUT, T_BOOT_EMERGENCY_HOURS, 24 * 7) # Example: 7 days total runtime
            for i in range(N_EM_GENS)
        ]
        # Pass step duration to Battery for energy calculations
        self.battery = Battery(
            MAX_BATTERY_CAPACITY, BATTERY_CHARGE_RATE, BATTERY_DISCHARGE_RATE,
            BATTERY_CHARGE_EFFICIENCY, BATTERY_DISCHARGE_EFFICIENCY, self.step_duration_hours
        )

        # Define Load Zones with different priorities and demand scaling
        self.load_zones = {
            'hi': LoadZone("HighPriority", scale=1.0, priority="high"),
            'lo1': LoadZone("LowPriority1", scale=0.7, priority="low1"),
            'lo2': LoadZone("LowPriority2", scale=0.5, priority="low2")
        }
        # Define the order for power distribution (highest priority first)
        self.load_zone_order = ['hi', 'lo1', 'lo2']

        # --- Observation Space Definition ---
        # Defines the structure of the information the agent receives each step.
        # Structure:
        # [current_time_normalized (0-1),
        #  battery_soc_normalized (0-1), battery_last_action (0,1,2),
        #  load_hi_demand, load_hi_shed (always 0),
        #  load_lo1_demand, load_lo1_shed (0/1),
        #  load_lo2_demand, load_lo2_shed (0/1),
        #  main_gen1_online (0/1), main_gen1_fail_timer, main_gen1_min_out, main_gen1_max_out,
        #  ..., (for N_MAIN_GENS)
        #  em_gen1_online (0/1), em_gen1_start_timer, em_gen1_runtime_left,
        #  ..., (for N_EM_GENS)
        # ]
        obs_dim = (
            1 + # Time
            2 + # Battery (SoC, Mode)
            len(self.load_zones) * 2 + # Loads (Demand, Shed Status per zone)
            N_MAIN_GENS * 4 + # Main Gens (Online, Timer, MinOut, MaxOut per gen)
            N_EM_GENS * 3 # Emergency Gens (Online, Timer, Runtime per gen)
        )
        # Using Box with broad bounds for simplicity. More specific bounds could be used
        # for features like timers or normalized values (e.g., [0, 1] for time/SoC).
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Action Space Definition ---
        # Defines the structure of the actions the agent can take.
        # Structure: A tuple/list with elements corresponding to:
        # [battery_mode (0:Idle, 1:Discharge, 2:Charge),
        #  em_gen1_mode (0:Idle, 1:Boot, 2:Shutdown),
        #  ..., (for N_EM_GENS)
        #  shed_lo1 (0:No Shed, 1:Shed),
        #  shed_lo2 (0:No Shed, 1:Shed)
        # ]
        action_space_list = [
            3, # Battery modes
            ] + [3] * N_EM_GENS + [ # Emergency generator modes (Idle/Boot/Shutdown)
            2, 2 # Shedding options for lo1 and lo2
            ]
        self.action_space = spaces.MultiDiscrete(action_space_list)

        # --- Internal State Variables for Logging/Rendering ---
        # These are reset/updated each step to track intermediate calculations.
        self._total_system_demand = 0.0 # Total demand including battery charge request (MW)
        self._total_available_power = 0.0 # Total power from all sources (Gen + Batt Discharge) (MW)
        self._power_balance_deficit = 0.0 # Unmet demand + un-charged battery need (MW)
        self._main_gen_failures_this_step = np.zeros(N_MAIN_GENS, dtype=bool) # Track failures for reward/info
        self._total_power_consumed_by_loads = 0.0 # Power actually consumed by loads (MW)
        self._power_consumed_for_charge = 0.0 # Power actually drawn for battery charge (MW)

        # Stores the info dictionary from the last step, used by render()
        self._last_info = {}


    def _get_obs(self):
        """
        Constructs the observation array from the current state of the environment components.

        This method gathers state information from time, battery, load zones,
        main generators, and emergency generators, formats it into a single
        NumPy array according to the defined `observation_space`.

        Returns:
            np.ndarray: The observation array (dtype=np.float32).
        """
        # 1. Normalized Time (0 to 1)
        obs = [self.current_time / self.episode_length_hours]

        # 2. Battery State (Normalized SoC, Last Action Mode)
        obs.extend(self.battery.get_state())

        # 3. Load Zone States (Demand, Shed Status for each zone in defined order)
        for zone_key in self.load_zone_order:
             obs.extend(self.load_zones[zone_key].get_state())

        # 4. Main Generator States (Online, Fail Timer, Min Output, Max Output for each)
        for gen in self.main_generators:
             obs.extend(gen.get_state())

        # 5. Emergency Generator States (Online, Start Timer, Runtime Left for each)
        for em_gen in self.emergency_generators:
             obs.extend(em_gen.get_state())

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state for a new episode.

        Sets time to 0, resets generator states (main online, emergency offline),
        sets battery SoC to 50%, resets load demands for time 0, and clears
        internal tracking variables.

        Args:
            seed (int | None): The seed to use for the environment's random number generator.
            options (dict | None): Additional options for resetting the environment (not used here).

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The initial observation after reset.
                - info (dict): An empty dictionary (or potentially containing initial state info).
        """
        super().reset(seed=seed) # Important for seeding RNG

        # --- Reset Time ---
        self.current_time = 0.0

        # --- Reset Components to Initial States ---
        # Battery: Start at 50% charge, idle state
        self.battery.soc = self.battery.max_capacity / 2.0
        self.battery.idle()
        # Main Generators: Start online, no failures pending
        for gen in self.main_generators:
            gen.online = True
            gen.fail_timer = 0
        # Emergency Generators: Start offline, not booting, full runtime
        for em_gen in self.emergency_generators:
            em_gen.online = False
            em_gen.start_timer = 0
            em_gen.runtime_left_steps = em_gen.total_runtime_steps
            em_gen._just_booted_this_step = False # Reset boot cost flag

        # --- Reset Load Zones ---
        # Update demand for time 0, ensure no shedding initially
        for zone in self.load_zones.values():
             zone.update_demand(self.current_time)
             zone.set_shed(False)

        # --- Reset Internal Logging/State Variables ---
        self._total_system_demand = 0.0
        self._total_available_power = 0.0
        self._power_balance_deficit = 0.0
        self._main_gen_failures_this_step[:] = False
        self._total_power_consumed_by_loads = 0.0
        self._power_consumed_for_charge = 0.0
        self._last_info = {} # Clear info from previous episode

        # Get the initial observation
        observation = self._get_obs()
        # Initial info dictionary (can be populated if needed)
        info = {}

        return observation, info

    def step(self, action):
        """
        Applies the agent's action and advances the environment simulation by one time step.

        This involves:
        1. Parsing the action.
        2. Updating the state of generators (failures, healing, booting, runtime).
        3. Updating load demands and applying shedding actions.
        4. Calculating total system demand (loads + potential battery charge).
        5. Calculating total available power (main gens, em gens, potential battery discharge).
        6. Distributing available power according to priorities (loads, then battery charge).
        7. Calculating the reward based on performance (meeting loads, costs, penalties).
        8. Advancing time and checking for episode termination.
        9. Constructing the next observation and info dictionary.
        10. Optionally rendering the state.

        Args:
            action (np.ndarray or tuple): The action chosen by the agent, conforming
                                          to the `action_space` definition.

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The observation after the step.
                - reward (float): The reward obtained during the step.
                - terminated (bool): True if the episode ended naturally (e.g., time limit).
                - truncated (bool): True if the episode ended prematurely (e.g., external limit, not used here).
                - info (dict): A dictionary containing auxiliary information about the step.
        """
        # Ensure the action is valid according to the defined action space
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # --- 1. Parse Action ---
        # Extract individual action components from the MultiDiscrete action tuple/array
        a_batt_mode = action[0] # 0:Idle, 1:Discharge, 2:Charge
        a_em_modes = action[1 : 1 + N_EM_GENS] # List of modes (0/1/2) for each EM gen
        a_shed_lo1 = action[1 + N_EM_GENS] # 0:No Shed, 1:Shed
        a_shed_lo2 = action[1 + N_EM_GENS + 1] # 0:No Shed, 1:Shed

        # --- 2. Reset Step-Specific State ---
        # Clear flags from the previous step
        self._main_gen_failures_this_step[:] = False

        # --- 3. Update Component States (pre-power-flow calculations) ---
        # Update Main Generators: Handle failures and healing timers
        for i, gen in enumerate(self.main_generators):
            if gen.update(): # update() returns True if a failure occurred this step
                self._main_gen_failures_this_step[i] = True

        # Update Emergency Generators: Process boot/shutdown commands and update timers/runtime
        for i, em_gen in enumerate(self.emergency_generators):
            em_action = a_em_modes[i]
            if em_action == 1: # Action: Command Boot
                 # command_boot attempts to start the boot sequence and sets an internal flag for cost
                 em_gen.command_boot()
            elif em_action == 2: # Action: Command Shutdown
                 em_gen.command_shutdown()
            # Update internal state (decrement timers, check runtime, transition online/offline)
            em_gen.update() # Handles timer countdowns and runtime decrement

        # Update Load Zones: Update demand based on time, apply agent's shedding actions
        for zone_key, zone in self.load_zones.items():
            zone.update_demand(self.current_time) # Update demand based on current time
            if zone_key == 'lo1':
                zone.set_shed(a_shed_lo1) # Apply shedding action
            elif zone_key == 'lo2':
                zone.set_shed(a_shed_lo2) # Apply shedding action
            # High priority zone cannot be shed by agent action
            if zone_key == 'hi':
                 zone.set_shed(False)


        # --- 4. Calculate Total System Demand ---

        # 4a. Calculate total effective demand from loads (demand after shedding is applied)
        total_effective_load_demand = sum(
            zone.get_effective_demand() for zone in self.load_zones.values()
        ) # MW

        # 4b. Determine power requested for battery charging, if actioned
        desired_power_for_battery_charge = 0.0
        if a_batt_mode == 2: # Action: Charge Battery
             # Battery will request power up to its charge rate limit
             desired_power_for_battery_charge = self.battery.charge_rate # MW

        # 4c. Total demand the system *needs* to meet this step
        self._total_system_demand = total_effective_load_demand + desired_power_for_battery_charge # MW


        # --- 5. Calculate Total Available Power ---

        # 5a. Power available from dispatchable sources (Emergency Gens, Battery Discharge)
        emergency_gen_power = sum(em_gen.get_current_output() for em_gen in self.emergency_generators) # MW

        battery_discharge_power = 0.0
        if a_batt_mode == 1: # Action: Discharge Battery
             # attempt_discharge calculates and returns power delivered to grid, updates SoC
             battery_discharge_power = self.battery.attempt_discharge() # MW
        else:
             # If not discharging, ensure battery is set to idle state internally
             self.battery.idle()

        # 5b. Calculate power required *from* Main Generators
        # This is the remaining demand after accounting for EM gen and battery discharge.
        # Can be negative if EM/Battery provide surplus.
        required_from_main_gens = self._total_system_demand - emergency_gen_power - battery_discharge_power # MW

        # 5c. Calculate actual output from Main Generators
        # Online main gens collectively try to meet the 'required_from_main_gens'.
        # They produce at least their collective minimum, up to their collective maximum.
        online_main_gens = [gen for gen in self.main_generators if gen.online]
        num_online_main_gens = len(online_main_gens)
        collective_min_main = sum(gen.min_output for gen in online_main_gens) # MW
        collective_max_main = sum(gen.max_output for gen in online_main_gens) # MW

        main_gen_actual_output = 0.0
        if num_online_main_gens > 0:
             # Ensure output is within [collective_min, collective_max].
             # If required is less than min, they still produce min (potential surplus).
             # If required is more than max, they produce max.
             main_gen_actual_output = np.clip(required_from_main_gens, collective_min_main, collective_max_main)
             # Correction: If required is negative (surplus from EM/Batt), main gens should still produce their minimum.
             main_gen_actual_output = max(collective_min_main, main_gen_actual_output)


        # 5d. Calculate total available power from *all* sources for this step
        self._total_available_power = main_gen_actual_output + emergency_gen_power + battery_discharge_power # MW

        # 5e. Calculate the theoretical power deficit *before* distribution
        # This represents the gap between total system demand and total available power.
        self._power_balance_deficit = max(0.0, self._total_system_demand - self._total_available_power) # MW


        # --- 6. Distribute Available Power (Priority Order: Loads -> Battery Charge) ---

        power_remaining_for_distribution = self._total_available_power # Start with all available power

        # 6a. Distribute to High Priority Load Zone
        hi_zone = self.load_zones['hi']
        # attempt_meet_demand returns power consumed and updates zone's internal 'met' status
        power_consumed_hi = hi_zone.attempt_meet_demand(power_remaining_for_distribution)
        power_remaining_for_distribution -= power_consumed_hi

        # 6b. Distribute to Low Priority 1 Load Zone
        lo1_zone = self.load_zones['lo1']
        power_consumed_lo1 = lo1_zone.attempt_meet_demand(power_remaining_for_distribution)
        power_remaining_for_distribution -= power_consumed_lo1

        # 6c. Distribute to Low Priority 2 Load Zone
        lo2_zone = self.load_zones['lo2']
        power_consumed_lo2 = lo2_zone.attempt_meet_demand(power_remaining_for_distribution)
        power_remaining_for_distribution -= power_consumed_lo2

        # Track total power consumed by all loads
        self._total_power_consumed_by_loads = power_consumed_hi + power_consumed_lo1 + power_consumed_lo2

        # 6d. Distribute remaining power to Battery Charging (if actioned)
        self._power_consumed_for_charge = 0.0 # Reset tracker for this step
        if a_batt_mode == 2: # Action: Charge Battery
             # attempt_charge takes available power, returns power actually drawn, updates SoC
             self._power_consumed_for_charge = self.battery.attempt_charge(available_grid_power=power_remaining_for_distribution)
             # Note: power_remaining_for_distribution is not strictly needed after this,
             # but could be tracked as 'surplus' or 'curtailed' power if desired.
             # power_remaining_for_distribution -= self._power_consumed_for_charge


        # --- 7. Calculate Reward ---
        # Combine rewards for meeting demand and costs/penalties for actions/events.
        reward = 0.0

        # Rewards for meeting load demands (check status after distribution)
        if hi_zone.was_met():
            reward += W_HI # Add reward for meeting high priority
        else:
            # Apply large penalty if high priority demand was not met
            # This check is crucial as hi_zone cannot be shed by action.
            if hi_zone.get_effective_demand() > 1e-6: # Only penalize if there was actual demand
                 reward -= C_UNMET_HI

        if lo1_zone.was_met():
            reward += W_LO1 # Add reward for meeting low priority 1
        if lo2_zone.was_met():
            reward += W_LO2 # Add reward for meeting low priority 2

        # Penalties for shedding low priority loads (applies if action was taken, regardless of met status)
        if lo1_zone.was_shed():
            reward -= C_SHED
        if lo2_zone.was_shed():
            reward -= C_SHED

        # Costs associated with Battery actions (based on energy transferred)
        # get_energy_costs() retrieves MWh calculated during attempt_charge/discharge
        energy_drawn_for_charge, energy_discharged = self.battery.get_energy_costs() # MWh
        reward -= C_BATT_DISCHARGE * energy_discharged # Cost for discharging
        reward -= C_BATT_CHARGE * energy_drawn_for_charge # Cost for charging (drawing from grid)

        # Costs associated with Emergency Generators
        for i, em_gen in enumerate(self.emergency_generators):
            # One-time cost for initiating boot sequence (flag set in command_boot)
            if em_gen._just_booted_this_step:
                 reward -= C_EM_BOOT

            # Running cost per hour when online
            if em_gen.online:
                 reward -= C_EM_RUN * self.step_duration_hours

                 # Optional: Cost for being online but idle (system had surplus power)
                 # Check if total power used was less than total available power
                 total_power_used = self._total_power_consumed_by_loads + self._power_consumed_for_charge
                 # Use a small tolerance for float comparison
                 if self._total_available_power > total_power_used + 1e-6:
                      reward -= C_EM_IDLE_ONLINE * self.step_duration_hours


        # Penalty for Main Generator failures
        reward -= np.sum(self._main_gen_failures_this_step) * C_FAIL


        # --- 8. Advance Time ---
        self.current_time += self.step_duration_hours

        # --- 9. Check Termination Conditions ---
        # Episode terminates if the simulation duration is reached.
        terminated = self.current_time >= self.episode_length_hours
        # Truncated is used for premature termination (e.g., time limits in wrappers), not used here.
        truncated = False

        # --- 10. Construct Info Dictionary ---
        # Provides auxiliary information about the step, useful for debugging and analysis.
        info = {
            # Timing
            'current_time': self.current_time,
            # Power Balance Summary
            'total_demand_MW': self._total_system_demand, # Includes load + charge request
            'available_power_MW': self._total_available_power, # From all sources
            'power_balance_deficit_MW': self._power_balance_deficit, # Demand - Available (if > 0)
            'power_consumed_loads_MW': self._total_power_consumed_by_loads,
            'power_consumed_charge_MW': self._power_consumed_for_charge,
            # Load Details
            'loads_demand_MW': {zone_key: zone._current_demand for zone_key, zone in self.load_zones.items()}, # Raw demand
            'loads_effective_demand_MW': {zone_key: zone.get_effective_demand() for zone_key, zone in self.load_zones.items()}, # Demand after shedding
            'loads_met': {zone_key: zone.was_met() for zone_key, zone in self.load_zones.items()}, # Met status
            'loads_shed': {'lo1': lo1_zone.was_shed(), 'lo2': lo2_zone.was_shed()}, # Shedding action status
            # Battery Details
            'battery_soc_MWh': self.battery.soc,
            'battery_action_mode': a_batt_mode, # Agent's action for battery
            'energy_drawn_for_charge_MWh': energy_drawn_for_charge, # For cost calc
            'energy_discharged_MWh': energy_discharged, # For cost calc
            # Generator Details
            'main_gen_online': [gen.online for gen in self.main_generators],
            'main_gen_failures_this_step': self._main_gen_failures_this_step.tolist(), # Convert numpy bool array
            'emergency_gens_online': [em_gen.online for em_gen in self.emergency_generators],
            'emergency_gens_booting': [em_gen.start_timer > 0 for em_gen in self.emergency_generators],
            'emergency_gens_runtime_left': [em_gen.runtime_left_steps for em_gen in self.emergency_generators],
            # Critical Failure Flag
            'critical_failure': not hi_zone.was_met() and hi_zone.get_effective_demand() > 1e-6 # True if HI priority unmet
        }

        # Store the info dict for potential use by the render() method
        self._last_info = info

        # --- 11. Construct Next Observation ---
        observation = self._get_obs()

        # --- 12. Render (if mode is set) ---
        if self.render_mode is not None:
             # Default sleep time for 'human' mode rendering can be passed here
             self.render()


        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Renders the current state of the environment.

        Uses the `_last_info` dictionary populated by the `step` method.
        Supports 'human' mode (formatted dashboard, clears screen) and
        'terminal' mode (prints step summary sequentially - handled in main.py).
        The actual rendering logic here is primarily for 'human' mode.

        Args:
            RENDER_SLEEP_TIME (float): Time in seconds to pause after rendering,
                                       primarily for 'human' mode visibility.
        """
        if self.render_mode is None:
            # Standard warning if render is called without a mode set
            gym.logger.warn(
                "You are calling render method without specifying any render_mode "
                "during environment initialization. Set `render_mode={'human'|'terminal'}` "
                "passed to the environment constructor."
            )
            return

        elif self.render_mode == 'human':
            # Use the stored info dictionary from the last step
            info = self._last_info
            if not info:
                 # Handle case where render is called before the first step or after reset
                 print("Render called but no step information available yet.")
                 return

            # --- Human Mode Rendering: Clear screen and print dashboard ---
            os.system("cls" if os.name == "nt" else "clear") # Clear terminal screen
            print("=" * 70)
            print(f"Time: {info.get('current_time', self.current_time):.1f} / {self.episode_length_hours:.1f} hours | Step Duration: {self.step_duration_hours:.1f} hr")
            print("-" * 70)

            # Power Balance Section
            print("📊 Power Balance:")
            print(f"  Demand (Loads + Charge Req): {info.get('total_demand_MW', 0.0):>6.1f} MW")
            print(f"  Available Power (Supply):    {info.get('available_power_MW', 0.0):>6.1f} MW")
            deficit = info.get('power_balance_deficit_MW', 0.0)
            deficit_color = "\033[91m" if deficit > 1e-3 else "\033[92m" # Red if deficit, Green otherwise
            print(f"  Deficit (Unmet Demand):    {deficit_color}{deficit:>6.1f} MW\033[0m") # Reset color
            print(f"  Power Consumed (Loads):      {info.get('power_consumed_loads_MW', 0.0):>6.1f} MW")
            print(f"  Power Consumed (Charging):   {info.get('power_consumed_charge_MW', 0.0):>6.1f} MW")
            print("-" * 70)

            # Load Zones Section
            print("🏘️ Load Zones:")
            loads_demand = info.get('loads_demand_MW', {})
            loads_effective = info.get('loads_effective_demand_MW', {})
            loads_met = info.get('loads_met', {})
            loads_shed = info.get('loads_shed', {})
            for zone_key in self.load_zone_order:
                zone = self.load_zones[zone_key]
                demand = loads_demand.get(zone_key, 0.0)
                effective = loads_effective.get(zone_key, 0.0)
                is_met = loads_met.get(zone_key, False)
                is_shed = loads_shed.get(zone_key, False) if zone_key in loads_shed else False

                status_str = ""
                if is_shed:
                    status_str = "\033[93m[ SHED ]\033[0m" # Yellow
                elif is_met:
                    status_str = "\033[92m[ MET ]\033[0m" # Green
                elif demand > 1e-6 : # Only show UNMET if there was demand
                    status_str = "\033[91m[UNMET!]\033[0m" # Red
                else:
                    status_str = "[ Idle ]" # No demand

                print(f"  {zone.name:<15} ({zone.priority}): Demand={demand:>5.1f} MW | Effective={effective:>5.1f} MW | Status: {status_str}")
            if info.get('critical_failure', False):
                 print("  \033[91m*** CRITICAL FAILURE: High Priority Load UNMET! ***\033[0m")
            print("-" * 70)

            # Battery Section
            print("🔋 Battery:")
            mode_map = {0: "Idle", 1: "Discharging", 2: "Charging"}
            batt_mode_action = info.get('battery_action_mode', -1)
            soc_mwh = info.get('battery_soc_MWh', self.battery.soc)
            max_cap = self.battery.max_capacity
            soc_perc = (soc_mwh / max_cap * 100) if max_cap > 0 else 0.0
            # Simple progress bar for SoC
            bar_length = 20
            filled_length = int(bar_length * soc_mwh / max_cap) if max_cap > 0 else 0
            soc_bar = '█' * filled_length + '-' * (bar_length - filled_length)

            print(f"  SoC: {soc_mwh:>6.1f} / {max_cap:.1f} MWh ({soc_perc:>3.0f}%) |{soc_bar}|")
            print(f"  Mode (Action): {mode_map.get(batt_mode_action, 'Unknown'):<12}")
            print(f"  Energy Flow: Charged={info.get('energy_drawn_for_charge_MWh', 0.0):>5.1f} MWh | Discharged={info.get('energy_discharged_MWh', 0.0):>5.1f} MWh")
            print("-" * 70)

            # Generators Section
            print("⚡ Generators:")
            # Main Generators
            main_online = info.get('main_gen_online', [False]*N_MAIN_GENS)
            main_failed = info.get('main_gen_failures_this_step', [False]*N_MAIN_GENS)
            online_main_count = sum(main_online)
            print(f"  --- Main ({online_main_count}/{N_MAIN_GENS} Online) ---")
            for i, gen in enumerate(self.main_generators):
                status = ""
                color = "\033[0m" # Default color
                if main_online[i]:
                    status = " Online"
                    color = "\033[92m" # Green
                else:
                    status = f" Offline (Heal: {gen.fail_timer}/{gen.heal_time_steps})"
                    color = "\033[90m" # Grey
                fail_msg = ""
                if main_failed[i]:
                    fail_msg = " \033[91m*FAILED*\033[0m" # Red
                    color = "\033[91m" # Make whole line red if failed this step
                print(f"  {color}{gen.name:<10}:{status:<25}{fail_msg}\033[0m")

            # Emergency Generators
            em_online = info.get('emergency_gens_online', [False]*N_EM_GENS)
            em_booting = info.get('emergency_gens_booting', [False]*N_EM_GENS)
            em_runtime = info.get('emergency_gens_runtime_left', [0]*N_EM_GENS)
            online_em_count = sum(em_online)
            print(f"  --- Emergency ({online_em_count}/{N_EM_GENS} Online) ---")
            for i, em_gen in enumerate(self.emergency_generators):
                status = ""
                color = "\033[0m" # Default color
                if em_online[i]:
                    status = f" Online (Out: {em_gen.output_power:.0f} MW)"
                    color = "\033[92m" # Green
                elif em_booting[i]:
                    status = f" Booting ({em_gen.start_timer}/{em_gen.boot_time_steps} left)"
                    color = "\033[93m" # Yellow
                else:
                    status = " Offline"
                    color = "\033[90m" # Grey
                runtime_color = "\033[91m" if em_runtime[i] < em_gen.total_runtime_steps * 0.1 else "\033[0m" # Red if low runtime
                print(f"  {color}{em_gen.name:<10}:{status:<25}\033[0m | Runtime Left: {runtime_color}{em_runtime[i]:>4}/{em_gen.total_runtime_steps}\033[0m steps")
            print("=" * 70)

            # Pause for visibility in human mode
            if self.RENDER_SLEEP_TIME > 0:
                time.sleep(self.RENDER_SLEEP_TIME)

    def close(self):
        """
        Performs any necessary cleanup.

        Currently, no specific cleanup actions are required for this environment.
        """
        print("Closing PowerGridEnv.")
        pass
