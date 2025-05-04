"""
Utility Classes and Functions for the Power Grid Environment.

This module contains helper components used by the main PowerGridEnv class,
including models for generators (main and emergency), the battery storage system,
load zones, and the demand profile function.
"""

import numpy as np

def demand_profile(t, scale=1.0):
    """
    Generates a scaled power demand based on the time of day.

    Simulates a typical daily demand curve with morning and evening peaks.
    The base demand is scaled by the provided factor and multiplied by 100
    to represent demand in Megawatts (MW).

    Args:
        t (float): The current time in hours (typically within a 0-24 hour cycle).
        scale (float): A scaling factor applied to the base demand profile.

    Returns:
        float: The calculated power demand in MW for the given time and scale.
    """
    t_wrapped = t % 24 # Ensure time wraps around every 24 hours
    base = 0.5 # Base load factor
    # Morning peak centered around 8:00
    morning_peak = 0.4 * np.exp(-(t_wrapped - 8)**2 / (2 * 2**2))
    # Evening peak centered around 19:00
    evening_peak = 0.6 * np.exp(-(t_wrapped - 19)**2 / (2 * 2**2))
    # Combine base, peaks, apply scale, and convert to MW
    return scale * (base + morning_peak + evening_peak) * 100 # Scale factor applied


class MainGenerator:
    """
    Represents a main power generator with operational limits and failure dynamics.

    Attributes:
        name (str): Identifier for the generator.
        min_output (float): Minimum power output (MW) when online.
        max_output (float): Maximum power output (MW) when online.
        fail_prob_per_step (float): Probability of failure during one time step.
        heal_time_steps (int): Number of time steps required to heal after failure.
        online (bool): Current status, True if operational, False if failed/healing.
        fail_timer (int): Countdown timer for healing (steps remaining).
    """
    def __init__(self, name, min_output, max_output, fail_prob_per_hour, heal_time_hours, ramp_rate = 20):
        """
        Initializes a MainGenerator instance.

        Args:
            name (str): The name of the generator.
            min_output (float): Minimum power output (MW).
            max_output (float): Maximum power output (MW).
            fail_prob_per_hour (float): Hourly failure probability.
            heal_time_hours (int): Healing time in hours.
        """
        self.name = name
        self.min_output = min_output # MW
        self.max_output = max_output # MW
        self.fail_prob_per_step = fail_prob_per_hour # Assumes 1 step = 1 hour
        self.heal_time_steps = heal_time_hours # Assumes 1 step = 1 hour
        self.ramp_rate = ramp_rate # Example: Can change output by 20 MW per step (assuming 1 hour steps)
        self._current_actual_output = 0.0 # Track actual output

        self.online = True # Starts online
        self.fail_timer = 0 # Starts with no failure pending

    def update(self):
        """
        Updates the generator's state for one time step.

        Handles the healing process if offline.
        Checks for random failure if online.

        Returns:
            bool: True if a failure occurred during this step, False otherwise.
        """
        failed_this_step = False
        if not self.online:
            # If offline, decrement heal timer
            self.fail_timer -= 1
            if self.fail_timer <= 0:
                # Healing complete, bring back online
                self.online = True
                self.fail_timer = 0
        else:
            # If online, check for failure
            if np.random.rand() < self.fail_prob_per_step:
                self.online = False
                self.fail_timer = self.heal_time_steps # Start healing timer
                failed_this_step = True # Report failure
        return failed_this_step

    def get_state(self):
        """
        Returns the current state of the generator for the environment observation.

        Returns:
            list: A list containing [online_status (float 0.0 or 1.0),
                     fail_timer (int), min_output (float), max_output (float)].
        """
        return [float(self.online), self.fail_timer, self.min_output, self.max_output]

    def set_desired_output(self, desired_output):
         """Sets the target output for this step (system's need)."""
         self._desired_output = np.clip(desired_output, self.min_output, self.max_output)
         # If offline, desired output is effectively 0
         if not self.online:
              self._desired_output = 0.0

    def get_actual_output(self, step_duration_hours):
         """Calculates and returns the actual output for this step, applying ramp rate."""
         if not self.online:
              self._current_actual_output = 0.0 # If offline, output is 0
              return 0.0

         # Calculate maximum possible change this step
         max_change = self.ramp_rate * step_duration_hours

         # Determine the target output we're trying to reach (already set by set_desired_output)
         target = self._desired_output

         # Calculate the difference between target and current output
         difference = target - self._current_actual_output

         # Determine how much we can actually change this step, limited by ramp rate
         change_amount = np.clip(difference, -max_change, max_change)

         # Update the current actual output
         self._current_actual_output += change_amount

         # Ensure output stays within the generator's min/max limits (even while ramping)
         # This might need careful thought - typically ramp limits apply *between* steps,
         # but the final output must be within min/max. Let's clip here.
         self._current_actual_output = np.clip(self._current_actual_output, self.min_output, self.max_output)

         return self._current_actual_output # Return the calculated output for this step
class EmergencyGenerator:
    """
    Represents an emergency generator with startup time and limited runtime.

    Attributes:
        name (str): Identifier for the generator.
        output_power (float): Fixed power output (MW) when online.
        boot_time_steps (int): Time steps required to start up.
        total_runtime_steps (int): Maximum number of steps the generator can run in total.
        online (bool): Current status, True if running, False otherwise.
        start_timer (int): Countdown timer for booting (steps remaining).
        runtime_left_steps (int): Remaining operational steps available.
        _just_booted_this_step (bool): Internal flag to track boot command for cost calculation.
    """
    def __init__(self, name, output_power, boot_time_hours, total_runtime_hours):
        """
        Initializes an EmergencyGenerator instance.

        Args:
            name (str): The name of the generator.
            output_power (float): Fixed power output (MW) when online.
            boot_time_hours (int): Startup time in hours.
            total_runtime_hours (int): Total available runtime in hours.
        """
        self.name = name
        self.output_power = output_power # MW
        self.boot_time_steps = boot_time_hours # Assumes 1 step = 1 hour
        self.total_runtime_steps = total_runtime_hours # Assumes 1 step = 1 hour

        self.online = False # Starts offline
        self.start_timer = 0 # Starts not booting
        self.runtime_left_steps = self.total_runtime_steps # Starts with full runtime
        self._just_booted_this_step = False # Internal flag for boot cost in the current step

    def command_boot(self):
        """
        Initiates the boot sequence if the generator is offline, not already booting,
        and has runtime remaining.

        Sets an internal flag `_just_booted_this_step` if the command is accepted,
        used for applying the boot cost in the environment step.

        Returns:
            bool: True if the boot command was accepted, False otherwise.
        """
        self._just_booted_this_step = False # Reset flag from any previous step's command
        if not self.online and self.start_timer == 0 and self.runtime_left_steps > 0:
            self.start_timer = self.boot_time_steps # Start boot timer
            self._just_booted_this_step = True # Set flag for reward calculation
            return True # Command accepted
        # Command ignored if already online, already booting, or out of runtime
        return False

    def command_shutdown(self):
        """
        Manually shuts down the generator if it is currently online.

        Returns:
            bool: True if the shutdown command was accepted (generator was online),
                  False otherwise (generator was already offline).
        """
        if self.online:
            self.online = False
            self.start_timer = 0 # Ensure boot timer is reset if it was somehow active
            return True # Command accepted
        # Command ignored if already offline
        return False

    def update(self):
        """
        Updates the generator's state for one time step.

        Handles the boot timer countdown and decrements remaining runtime if online.
        Automatically shuts down if runtime expires.
        """
        # Note: _just_booted_this_step is set by command_boot, not modified here.

        # Handle booting process
        if self.start_timer > 0:
            self.start_timer -= 1
            if self.start_timer == 0:
                # Boot sequence finished, generator comes online
                self.online = True
                # Runtime consumption starts from the next step where it's online

        # Handle runtime consumption if online
        if self.online:
            if self.runtime_left_steps > 0:
                self.runtime_left_steps -= 1 # Consume one step of runtime
                if self.runtime_left_steps == 0:
                    # Ran out of fuel/runtime, automatically shut down
                    self.online = False
            else:
                 # Should not happen if logic is correct, but ensures it goes offline if runtime is 0
                 self.online = False


    def get_current_output(self):
        """
        Returns the current power output of the generator.

        Returns:
            float: The generator's output power (MW) if online, otherwise 0.0.
        """
        return self.output_power if self.online else 0.0

    def get_state(self):
        """
        Returns the current state of the generator for the environment observation.

        Returns:
            list: A list containing [online_status (float 0.0 or 1.0),
                     start_timer (int), runtime_left_steps (int)].
        """
        return [float(self.online), self.start_timer, self.runtime_left_steps]

class Battery:
    """
    Represents a battery energy storage system (BESS).

    Handles charging and discharging dynamics, including efficiency losses
    and state of charge (SoC) tracking.

    Attributes:
        max_capacity (float): Maximum energy storage capacity (MWh).
        charge_rate (float): Maximum power draw rate during charging (MW).
        discharge_rate (float): Maximum power injection rate during discharging (MW).
        charge_efficiency (float): Efficiency of storing energy (0 to 1).
        discharge_efficiency (float): Efficiency of delivering energy (0 to 1).
        step_duration_hours (float): Duration of a single environment time step in hours.
        soc (float): Current state of charge (energy stored) (MWh).
        _last_action_mode (int): Records the last action taken (0:Idle, 1:Discharge, 2:Charge).
        _energy_drawn_this_step (float): Energy drawn from grid for charging in the last step (MWh).
        _energy_discharged_this_step (float): Energy delivered to grid from discharging in the last step (MWh).
    """
    def __init__(self, max_capacity, charge_rate, discharge_rate, charge_efficiency, discharge_efficiency, step_duration_hours):
        """
        Initializes the Battery instance.

        Args:
            max_capacity (float): Maximum storage capacity (MWh).
            charge_rate (float): Maximum charging power (MW).
            discharge_rate (float): Maximum discharging power (MW).
            charge_efficiency (float): Charging efficiency factor.
            discharge_efficiency (float): Discharging efficiency factor.
            step_duration_hours (float): Duration of one time step in hours.
        """
        self.max_capacity = max_capacity # MWh
        self.charge_rate = charge_rate # MW
        self.discharge_rate = discharge_rate # MW
        self.charge_efficiency = charge_efficiency # Dimensionless
        self.discharge_efficiency = discharge_efficiency # Dimensionless
        self.step_duration_hours = step_duration_hours # hours

        self.soc = max_capacity / 2.0 # Start at 50% SoC (MWh)
        self._last_action_mode = 0 # 0=Idle, 1=Discharge, 2=Charge
        self._energy_drawn_this_step = 0.0 # MWh (for charge cost calculation)
        self._energy_discharged_this_step = 0.0 # MWh (for discharge cost/benefit calculation)
        self._energy_stored_this_step = 0.0

    def attempt_discharge(self):
        """
        Attempts to discharge the battery based on its discharge rate and current SoC.

        Calculates the actual power delivered to the grid, considering the discharge
        rate limit, available energy (SoC), step duration, and discharge efficiency.
        Updates the SoC accordingly. Records the energy delivered for cost calculation.

        Returns:
            float: The actual power delivered to the grid (MW) during this step.
                   This value is limited by rate, SoC, and efficiency.
        """
        self._last_action_mode = 1 # Record action
        self._energy_drawn_this_step = 0.0 # Reset charging energy tracker
        self._energy_stored_this_step = 0.0 # Reset for discharge step

        # 1. Max power the battery *hardware* can push out (MW)
        max_battery_power_rate = self.discharge_rate

        # 2. Max energy available in the battery (MWh)
        max_energy_by_soc = self.soc

        # 3. Max power the battery can *sustain* for the step duration based on available energy (MW)
        # If step duration is 0, avoid division by zero.
        max_power_by_soc = max_energy_by_soc / self.step_duration_hours if self.step_duration_hours > 0 else 0.0

        # 4. Determine the gross power *removed* from the battery (MW)
        # Limited by both the hardware rate and the sustainable rate based on SoC.
        actual_discharge_power_gross = min(max_battery_power_rate, max_power_by_soc)

        # 5. Calculate the power *delivered* to the grid after efficiency loss (MW)
        actual_discharge_power_to_grid = actual_discharge_power_gross * self.discharge_efficiency

        # 6. Calculate the actual energy *removed* from the battery's internal storage (MWh)
        actual_discharge_energy_gross = actual_discharge_power_gross * self.step_duration_hours

        # 7. Update SoC by removing the gross energy
        self.soc -= actual_discharge_energy_gross
        self.soc = max(0.0, self.soc) # Ensure SoC doesn't drop below zero

        # 8. Record the energy *delivered* to the grid for cost/reward calculation (MWh)
        self._energy_discharged_this_step = actual_discharge_power_to_grid * self.step_duration_hours

        # Return the power delivered to the grid
        return actual_discharge_power_to_grid

    def attempt_charge(self, available_grid_power):
        """
        Attempts to charge the battery using available power from the grid.

        Calculates the actual power drawn from the grid, considering the charge
        rate limit, available grid power, remaining storage capacity (headroom),
        step duration, and charge efficiency. Updates the SoC accordingly.
        Records the energy drawn for cost calculation.

        Args:
            available_grid_power (float): Power available from the grid (MW)
                                          that can potentially be used for charging.

        Returns:
            float: The actual power drawn from the grid (MW) for charging during this step.
                   This value is limited by rate, capacity, efficiency, and available grid power.
        """
        self._last_action_mode = 2 # Record action
        self._energy_discharged_this_step = 0.0 # Reset discharging energy tracker
        self._energy_stored_this_step = 0.0 # Reset for charge step

        # 1. Max power the battery *hardware* can draw (MW)
        max_battery_draw_rate = self.charge_rate

        # 2. Max energy the battery can *store* before reaching full capacity (MWh)
        headroom_energy = self.max_capacity - self.soc

        # 3. Max energy the battery *needs to draw* from the grid to fill the headroom, considering efficiency (MWh)
        # If efficiency is 0, treat as infinite draw needed (or handle as error).
        max_draw_energy_needed = headroom_energy / self.charge_efficiency if self.charge_efficiency > 0 else float('inf')

        # 4. Max power the battery can *sustainably draw* for the step duration to fill headroom (MW)
        # If step duration is 0, avoid division by zero.
        max_power_by_capacity = max_draw_energy_needed / self.step_duration_hours if self.step_duration_hours > 0 else float('inf')

        # 5. Determine the desired power *draw* from the grid (MW)
        # Limited by both the hardware rate and the sustainable rate based on capacity.
        desired_draw_power = min(max_battery_draw_rate, max_power_by_capacity)

        # 6. Determine the actual power *drawn* from the grid (MW)
        # Limited by the desired draw and the power actually available from the grid.
        actual_draw_power = min(desired_draw_power, available_grid_power)

        # 7. Calculate the actual energy *drawn* from the grid (MWh)
        actual_draw_energy = actual_draw_power * self.step_duration_hours

        # 8. Calculate the actual energy *stored* in the battery after efficiency loss (MWh)
        actual_stored_energy = actual_draw_energy * self.charge_efficiency

        # 9. Update SoC by adding the stored energy
        self.soc += actual_stored_energy
        self.soc = min(self.max_capacity, self.soc) # Ensure SoC doesn't exceed maximum

        # 10. Record the energy *drawn* from the grid for cost calculation (MWh)
        self._energy_drawn_this_step = actual_draw_energy

        # --- Record energy stored for charging reward ---
        self._energy_stored_this_step = actual_stored_energy

        # Return the power drawn from the grid
        return actual_draw_power

    def idle(self):
        """Sets the battery mode to idle and resets step energy trackers."""
        self._last_action_mode = 0
        self._energy_drawn_this_step = 0.0
        self._energy_discharged_this_step = 0.0

    def get_state(self):
        """
        Returns the current state of the battery for the environment observation.

        Returns:
            list: A list containing [normalized_soc (float 0.0 to 1.0),
                     last_action_mode (float 0.0, 1.0, or 2.0)].
        """
        # Normalize SoC to be between 0 and 1 for the observation space
        normalized_soc = self.soc / self.max_capacity if self.max_capacity > 0 else 0.0
        return [normalized_soc, float(self._last_action_mode)]

    def get_energy_costs(self):
        """
        Returns the energy drawn and discharged in the last step for cost calculation.

        Returns:
            tuple: A tuple containing (energy_drawn_MWh, energy_discharged_MWh).
        """
        return self._energy_drawn_this_step, self._energy_discharged_this_step
    
    def get_energy_stored_this_step(self):
        """Returns the energy (MWh) actually stored in the battery in the last step."""
        return self._energy_stored_this_step


class LoadZone:
    """
    Represents a load zone with dynamic demand and controllable shedding.

    Attributes:
        name (str): Identifier for the load zone.
        scale (float): Scaling factor applied to the base demand profile.
        priority (str): Priority level ('high', 'low1', 'low2', etc.).
        _current_demand (float): Calculated demand (MW) for the current time step.
        _met (bool): Flag indicating if demand was fully met in the last power distribution phase.
        _shed (bool): Flag indicating if the zone was commanded to shed load in the last step.
    """
    def __init__(self, name, scale, priority="normal"):
        """
        Initializes a LoadZone instance.

        Args:
            name (str): The name of the load zone.
            scale (float): Scaling factor for the demand profile.
            priority (str): Priority level string (e.g., "high", "low").
        """
        self.name = name
        self.scale = scale # Scaling factor for base demand
        self.priority = priority # Priority level
        self._current_demand = 0.0 # MW, updated each step
        self._met = False # Was demand met in the last step?
        self._shed = False # Was the zone actively shed in the last step?

    def update_demand(self, current_time):
        """
        Updates the zone's demand based on the current simulation time.

        Args:
            current_time (float): The current time in hours.
        """
        self._current_demand = demand_profile(current_time, self.scale)

    def set_shed(self, shed_status):
        """
        Sets the shedding status based on the agent's action.

        Args:
            shed_status (bool or int): True or 1 to shed, False or 0 to not shed.
        """
        self._shed = bool(shed_status)

    def get_effective_demand(self):
        """
        Returns the demand that needs to be met by the grid.

        Returns 0 if the zone is currently shed, otherwise returns the
        calculated current demand.

        Returns:
            float: The effective power demand (MW) for this step.
        """
        return self._current_demand if not self._shed else 0.0

    def attempt_meet_demand(self, available_power):
        """
        Attempts to satisfy the zone's effective demand with the provided power.

        Updates the internal `_met` status flag based on whether the demand
        was fully satisfied. Consumes power from the available pool.

        Args:
            available_power (float): Power (MW) allocated to this zone (and potentially
                                     lower priority zones/charging).

        Returns:
            float: The amount of power (MW) actually consumed by this load zone.
        """
        # Reset met status at the beginning of the attempt for this step
        self._met = False

        if self._shed:
            # If shed, consumes no power and demand is considered unmet (unless demand was zero)
            self._met = (self._current_demand <= 1e-6) # Consider met if demand was effectively zero
            return 0.0

        # Demand that needs to be met this step
        demand_needed = self.get_effective_demand()

        # Power consumed is the minimum of what's needed and what's available
        power_consumed = min(demand_needed, available_power)

        # Check if the demand was fully met (within a small tolerance for float comparisons)
        if power_consumed >= demand_needed - 1e-6: # Use tolerance for robustness
             self._met = True
        # else: # Implicitly False from initialization
        #     self._met = False

        return power_consumed

    def was_met(self):
        """
        Returns whether the demand was fully met in the last power distribution phase.

        Returns:
            bool: True if demand was met, False otherwise.
        """
        return self._met

    def was_shed(self):
        """
        Returns whether the zone was commanded to shed load in the last step.

        Returns:
            bool: True if the zone was shed, False otherwise.
        """
        return self._shed

    def get_state(self):
        """
        Returns the current state of the load zone for the environment observation.

        Returns:
            list: A list containing [current_demand (float), shed_status (float 0.0 or 1.0)].
        """
        return [self._current_demand, float(self._shed)]
