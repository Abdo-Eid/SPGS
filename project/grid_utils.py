import numpy as np
import os
from numba import jit, float64, bool_, int64

# --- Numba Optimized Functions ---
@jit(float64(float64[:], bool_[:], int64), nopython=True, cache=True)
def compute_served_load(demands, blackout_flags, n_loads):
    """
    Calculates the total load served based on demand and blackout status.

    Args:
        demands (np.ndarray): Array of load demands.
        blackout_flags (np.ndarray): Boolean array indicating if a load is blacked out.
        n_loads (int): Number of loads.

    Returns:
        float: Total served load in MW.
    """
    total = 0.0
    for i in range(n_loads):
        if not blackout_flags[i]:
            total += demands[i]
    return total

@jit(int64[:](float64[:], float64[:], int64[:], int64), nopython=True, cache=True)
def check_overloads(stresses, capacities, edge_indices, n_edges):
    """
    Identifies overloaded edges based on stress and capacity.

    Args:
        stresses (np.ndarray): Array of power flow (stress) on each edge.
        capacities (np.ndarray): Array of capacities for each edge.
        edge_indices (np.ndarray): Original indices of the edges being checked.
        n_edges (int): Number of edges being checked.

    Returns:
        np.ndarray: Array of indices corresponding to the overloaded edges.
    """
    overloaded = np.zeros(n_edges, dtype=np.int64)
    count = 0
    for i in range(n_edges):
        # Check if capacity is finite and stress exceeds capacity
        if capacities[i] != np.inf and stresses[i] > capacities[i]:
            overloaded[count] = edge_indices[i]
            count += 1
    return overloaded[:count]

# --- Save/Load Q-table ---
Q_TABLE_FILENAME = 'power_grid_q_table_np.npz' # Default filename

def save_q_table(q_table, state_to_index, next_state_idx, filename=Q_TABLE_FILENAME):
    """
    Saves the Q-table and associated state mapping to a compressed NumPy file.

    Args:
        q_table (np.ndarray): The Q-table.
        state_to_index (dict): Dictionary mapping state tuples to row indices in Q-table.
        next_state_idx (int): The index for the next new state to be added.
        filename (str): The path to save the file to.
    """
    try:
        # Convert state tuples (keys) to strings for saving
        state_keys = np.array([str(k) for k in state_to_index.keys()])
        state_values = np.array(list(state_to_index.values()), dtype=np.int64)

        # Save the relevant part of the Q-table and the state mapping
        np.savez(filename,
                 q_table=q_table[:next_state_idx,:], # Save only used rows
                 state_keys=state_keys,
                 state_values=state_values,
                 next_state_index=np.array([next_state_idx]) # Save as single-element array
                )
        print(f"Q-table saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving Q-table: {e}")

def load_q_table(filename, action_space_size, state_length):
    """
    Loads the Q-table and state mapping from a file. Initializes if file not found or invalid.

    Args:
        filename (str): The path to load the file from.
        action_space_size (int): The number of possible actions (columns in Q-table).
        state_length (int): Expected length of the state tuple (used for validation, though not strictly here).

    Returns:
        tuple: (q_table, state_to_index, next_state_index)
               Returns initialized empty structures if loading fails.
    """
    initial_q_table_size = 1024 # Default size if creating a new table
    # Default empty structures to return on failure
    empty_q_table = np.zeros((initial_q_table_size, action_space_size), dtype=np.float32)
    empty_state_map = {}
    empty_next_idx = 0

    if not os.path.exists(filename):
        print(f"No saved Q-table found at {filename}. Initializing new table.")
        return empty_q_table, empty_state_map, empty_next_idx

    try:
        data = np.load(filename, allow_pickle=True)
        loaded_q_table = data['q_table']
        loaded_state_keys = data['state_keys']
        loaded_state_values = data['state_values']
        next_state_idx = int(data['next_state_index'][0]) # Extract integer index

        # --- Validation ---
        if loaded_q_table.ndim != 2 or loaded_q_table.shape[1] != action_space_size:
            print(f"Warning: Q-table shape mismatch (loaded {loaded_q_table.shape}, expected N x {action_space_size})! Re-initializing.")
            return empty_q_table, empty_state_map, empty_next_idx

        # --- Reconstruct State Mapping ---
        state_to_index = {}
        for k_str, v_idx in zip(loaded_state_keys, loaded_state_values):
            try:
                # Convert string representation back to tuple of integers
                state_tuple = tuple(map(int, k_str.strip('()').split(',')))
                state_to_index[state_tuple] = v_idx
            except ValueError:
                print(f"Warning: Could not parse state key '{k_str}'. Skipping.")
                continue

        # --- Ensure Q-table has sufficient capacity (pad if needed) ---
        # Determine required capacity (at least initial size, or loaded size)
        current_capacity = max(initial_q_table_size, loaded_q_table.shape[0])
        if current_capacity > loaded_q_table.shape[0]:
            # Pad the loaded table with zeros if its capacity is less than the initial size
            q_table = np.pad(loaded_q_table, ((0, current_capacity - loaded_q_table.shape[0]), (0, 0)), mode='constant')
        else:
            # Use the loaded table as is if its capacity is sufficient
            q_table = loaded_q_table

        print(f"Q-table loaded successfully from {filename} (Size: {q_table.shape}, States: {len(state_to_index)})")
        return q_table, state_to_index, next_state_idx

    except Exception as e:
        print(f"Error loading Q-table from {filename}: {e}. Returning empty Q-table.")
        return empty_q_table, empty_state_map, empty_next_idx
