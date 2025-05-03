import numpy as np

def demand_profile(t, scale=1.0):
    """Generates a scaled power demand based on time of day."""
    t_wrapped = t % 24
    base = 0.5
    morning_peak = 0.4 * np.exp(-(t_wrapped - 8)**2 / (2 * 2**2))
    evening_peak = 0.6 * np.exp(-(t_wrapped - 19)**2 / (2 * 2**2))
    return scale * (base + morning_peak + evening_peak) * 100 # Scale factor applied


class MainGenerator:
    """Represents a main power generator with failure dynamics."""
    def __init__(self, name, min_output, max_output, fail_prob_per_hour, heal_time_hours):
        self.name = name
        self.min_output = min_output
        self.max_output = max_output
        self.fail_prob_per_step = fail_prob_per_hour # Assuming 1 step = 1 hour
        self.heal_time_steps = heal_time_hours # Assuming 1 step = 1 hour

        self.online = True
        self.fail_timer = 0
        # Note: Main generators' output is determined by system need in env.step,
        # not by calling attempt_output here directly based on an agent action.

    # def attempt_output(self, requested_power): # This method is not used in env.step's current logic
    #     """Calculates potential output based on request, if online."""
    #     if not self.online:
    #         return 0.0
    #     return np.clip(requested_power, self.min_output, self.max_output)

    def update(self):
        """Updates generator state (failure/healing). Returns True if a failure occurred."""
        failed_this_step = False
        if not self.online:
            self.fail_timer -= 1
            if self.fail_timer <= 0:
                self.online = True
                self.fail_timer = 0
        else:
            # Check for failure only if online
            if np.random.rand() < self.fail_prob_per_step:
                self.online = False
                self.fail_timer = self.heal_time_steps
                failed_this_step = True
        return failed_this_step

    def get_state(self):
        """Returns the current state of the generator for observation."""
        return [float(self.online), self.fail_timer, self.min_output, self.max_output]

class EmergencyGenerator:
    """Represents an emergency generator with startup time and limited runtime."""
    def __init__(self, name, output_power, boot_time_hours, total_runtime_hours):
        self.name = name
        self.output_power = output_power
        self.boot_time_steps = boot_time_hours # Assuming 1 step = 1 hour
        self.total_runtime_steps = total_runtime_hours # Assuming 1 step = 1 hour

        self.online = False
        self.start_timer = 0 # Countdown to online
        self.runtime_left_steps = self.total_runtime_steps # Countdown of remaining runtime
        self._just_booted_this_step = False # Flag for boot cost

    def command_boot(self):
        """Initiates the boot sequence if not already online or booting."""
        self._just_booted_this_step = False # Reset flag from previous step command
        if not self.online and self.start_timer == 0 and self.runtime_left_steps > 0:
            self.start_timer = self.boot_time_steps
            self._just_booted_this_step = True # Set flag for reward in the current step
            return True # Command accepted
        return False # Command ignored (already online, booting, or out of runtime)

    def command_shutdown(self):
        """Shuts down the generator if online."""
        if self.online:
            self.online = False
            self.start_timer = 0 # Cancel boot if pending (shouldn't happen if online, but good practice)
            return True # Command accepted
        return False # Command ignored (already offline)


    def update(self):
        """Updates generator state (booting, online runtime)."""
        # _just_booted_this_step is handled in command_boot
        if self.start_timer > 0:
            self.start_timer -= 1
            if self.start_timer == 0:
                # Transition to online state
                self.online = True
                # Ensure we don't consume runtime in the very step it becomes online from timer reaching 0
                # This depends on whether update is called before or after power distribution logic.
                # If called before distribution, runtime should ideally start decrementing *after* outputting power.
                # Current structure calls update first, then distributes. Let's assume runtime is consumed *per step online*.
                # No change needed here based on this assumption.

        if self.online:
            if self.runtime_left_steps > 0:
                self.runtime_left_steps -= 1
                if self.runtime_left_steps == 0:
                    # Out of runtime, automatically go offline
                    self.online = False

    def get_current_output(self):
        """Returns the current power output of the generator."""
        return self.output_power if self.online else 0.0

    def get_state(self):
        """Returns the current state of the generator for observation."""
        return [float(self.online), self.start_timer, self.runtime_left_steps]

class Battery:
    """Represents a battery energy storage system."""
    def __init__(self, max_capacity, charge_rate, discharge_rate, charge_efficiency, discharge_efficiency, step_duration_hours):
        self.max_capacity = max_capacity
        self.charge_rate = charge_rate # MW
        self.discharge_rate = discharge_rate # MW
        self.charge_efficiency = charge_efficiency # dimensionless (energy_stored / energy_drawn)
        self.discharge_efficiency = discharge_efficiency # dimensionless (energy_delivered / energy_removed_from_battery)
        self.step_duration_hours = step_duration_hours

        self.soc = max_capacity / 2.0 # State of Charge (MWh)
        self._last_action_mode = 0 # For observation/info: 0=Idle, 1=Discharge, 2=Charge
        self._energy_drawn_this_step = 0.0 # MWh drawn from grid for charging (for cost)
        self._energy_discharged_this_step = 0.0 # MWh delivered to grid from discharging (for cost/benefit)

    def attempt_discharge(self):
        """Attempts to discharge battery. Returns power delivered to grid (MW)."""
        self._last_action_mode = 1
        self._energy_drawn_this_step = 0.0 # Reset for this step

        # Max power the battery can provide based on its rate limit
        max_battery_power_rate = self.discharge_rate # MW

        # Max energy that can be removed based on current SoC and step duration (MWh)
        max_energy_by_soc = self.soc # MWh

        # Max power rate the battery can sustain based on current SoC and step duration (MW)
        max_power_by_soc = max_energy_by_soc / self.step_duration_hours if self.step_duration_hours > 0 else 0.0

        # Gross power (power removed from battery) limited by rate and SoC
        actual_discharge_power_gross = min(max_battery_power_rate, max_power_by_soc)

        # Power delivered to the grid after accounting for discharge efficiency
        actual_discharge_power_to_grid = actual_discharge_power_gross * self.discharge_efficiency

        # Calculate actual energy removed from battery
        actual_discharge_energy_gross = actual_discharge_power_gross * self.step_duration_hours

        # Update SoC - Use gross energy removed from battery
        self.soc -= actual_discharge_energy_gross
        self.soc = max(0.0, self.soc) # Ensure SoC doesn't go below zero

        # Record energy delivered to grid for costing
        self._energy_discharged_this_step = actual_discharge_power_to_grid * self.step_duration_hours

        return actual_discharge_power_to_grid # Return power delivered to grid

    def attempt_charge(self, available_grid_power):
        """Attempts to charge battery given available grid power. Returns power drawn from grid (MW)."""
        self._last_action_mode = 2
        self._energy_discharged_this_step = 0.0 # Reset for this step

        # Max power battery can draw based on its rate limit
        max_battery_draw_rate = self.charge_rate # MW

        # Max energy battery can store until full (MWh)
        max_storage_energy = self.max_capacity - self.soc # MWh

        # Max energy battery can *draw* from the grid to reach max_storage_energy, accounting for efficiency
        max_draw_energy_by_capacity = max_storage_energy / self.charge_efficiency if self.charge_efficiency > 0 else float('inf') # MWh

        # Max power rate battery can draw based on remaining capacity and step duration (MW)
        max_power_by_capacity = max_draw_energy_by_capacity / self.step_duration_hours if self.step_duration_hours > 0 else float('inf') # MW

        # Desired power draw, limited by rate limit and remaining capacity
        desired_draw_power = min(max_battery_draw_rate, max_power_by_capacity) # MW

        # Actual power drawn, limited by desired draw and available grid power
        actual_draw_power = min(desired_draw_power, available_grid_power) # MW

        # Calculate actual energy drawn from grid
        actual_draw_energy = actual_draw_power * self.step_duration_hours # MWh

        # Calculate actual energy stored, accounting for charge efficiency
        actual_stored_energy = actual_draw_energy * self.charge_efficiency # MWh

        # Update SoC - Use energy actually stored
        self.soc += actual_stored_energy
        self.soc = min(self.max_capacity, self.soc) # Ensure SoC doesn't exceed max

        # Record energy drawn from grid for costing
        self._energy_drawn_this_step = actual_draw_energy

        return actual_draw_power # Return power drawn from grid


    def idle(self):
        """Sets battery mode to idle."""
        self._last_action_mode = 0
        self._energy_drawn_this_step = 0.0
        self._energy_discharged_this_step = 0.0

    def get_state(self):
        """Returns the current state of the battery for observation."""
        # Normalizing SoC is good practice for observation space
        return [self.soc / self.max_capacity, float(self._last_action_mode)]

    def get_energy_costs(self):
        """Returns energy drawn (charge) and energy discharged (discharge) in MWh for the last step."""
        return self._energy_drawn_this_step, self._energy_discharged_this_step


class LoadZone:
    """Represents a load zone with dynamic demand and shedding capability."""
    def __init__(self, name, scale, priority="normal"):
        self.name = name
        self.scale = scale # Scaling factor for the base demand profile
        self.priority = priority # Priority level (e.g., "high", "low1", "low2")
        self._current_demand = 0.0 # MW
        self._met = False # Flag indicating if demand was fully met this step
        self._shed = False # Flag indicating if the zone was commanded to shed this step

    def update_demand(self, current_time):
        """Updates the demand for the current time step."""
        self._current_demand = demand_profile(current_time, self.scale)

    def set_shed(self, shed_status):
        """Sets whether the load zone is shed (0=No, 1=Yes)."""
        self._shed = bool(shed_status)

    def get_effective_demand(self):
        """Returns the demand that needs to be met (0 if shed)."""
        return self._current_demand if not self._shed else 0.0

    def attempt_meet_demand(self, available_power):
        """Attempts to meet the demand given available power. Returns power consumed (MW)."""
        # Reset met status for this step
        self._met = False

        if self._shed:
            # If shed, demand is not met and no power is consumed
            return 0.0

        demand_needed = self._current_demand
        power_consumed = min(demand_needed, available_power)

        # If consumed power is equal to or greater than the demand, it's met
        # Using a small epsilon for floating point comparison robustness is sometimes done,
        # but direct comparison with >= is usually sufficient and clearer.
        if power_consumed >= demand_needed:
             self._met = True
        else:
            self._met = False # Partially met or not met at all

        return power_consumed

    def was_met(self):
        """Returns True if the demand for this zone was fully met in the last step."""
        return self._met

    def was_shed(self):
        """Returns True if this zone was commanded to shed in the last step."""
        return self._shed

    def get_state(self):
        """Returns the current state of the load zone for observation."""
        return [self._current_demand, float(self._shed)] # Return current demand and shed status