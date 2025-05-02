import networkx as nx
import numpy as np
from collections import defaultdict
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.components import connected_components

# Import Numba-optimized functions from grid_utils
from grid_utils import compute_served_load, check_overloads

def simulate_grid_state(graph):
    """
    Simulates the power grid state based on the current graph topology and demands.
    Determines served load, blacked-out zones, and overloaded lines.

    Args:
        graph (nx.DiGraph): The current state of the power grid graph,
                            including 'in_service' status for edges and
                            'current_demand_mw' for load nodes.

    Returns:
        tuple: (served_load_mw, blackout_status, overloaded_lines)
            - served_load_mw (float): Total load currently being served.
            - blackout_status (dict): {load_node: is_blacked_out (bool)}
            - overloaded_lines (set): Set of tuples (u, v) representing overloaded lines/transmission.
    """
    # Identify load nodes and initialize blackout status (all blacked out initially)
    load_nodes = [n for n, data in graph.nodes(data=True) if data['type'] == 'Load Zone']
    n_loads = len(load_nodes)
    blackout_status = {n: True for n in load_nodes} # Assume blackout initially

    # Handle empty graph case
    if graph.number_of_nodes() == 0:
        return 0.0, blackout_status, set()

    # --- Connectivity Check (Load Zones to Generators) ---

    # Identify active generators
    active_generators = [n for n, data in graph.nodes(data=True) if data['type'] == 'Generator']

    # Consider only 'in_service' transmission, tie lines, and regular lines for connectivity
    # Feeders are handled separately below.
    operating_edges = [(u, v) for u, v, data in graph.edges(data=True)
                       if data.get('in_service', True) and data.get('component') in ('transmission', 'tie_line', 'line')]

    # Create an undirected subgraph representing the operating transmission/line network
    # Using a copy prevents modification of the original graph structure implicitly
    operating_graph_undirected = graph.edge_subgraph(operating_edges).copy().to_undirected()

    # Handle case where operating graph might be empty
    if operating_graph_undirected.number_of_nodes() == 0:
         return 0.0, blackout_status, set() # No path possible if no operating nodes

    # Find connected components in the operating network
    components = list(connected_components(operating_graph_undirected))
    node_to_component = {} # Map each node to its component index
    gen_components = set() # Set of component indices that contain at least one generator

    for idx, comp in enumerate(components):
        has_generator = False
        for node in comp:
             node_to_component[node] = idx
             if node in active_generators:
                 has_generator = True
        if has_generator:
            gen_components.add(idx) # Mark this component as energized

    # Determine if each load zone can be served
    for load_node in load_nodes:
        is_connected = False
        # Find substations/nodes that feed this load zone via 'in_service' feeders
        feeder_sources = [u for u, v, data in graph.edges(data=True)
                          if v == load_node and data.get('in_service', True) and data.get('component') == 'feeder']

        for feeder_source_node in feeder_sources:
            # Check if the feeder's source node is part of an energized component
            if feeder_source_node in node_to_component and node_to_component[feeder_source_node] in gen_components:
                is_connected = True
                break # Found a path via at least one feeder

        # Update blackout status based on connectivity
        if is_connected:
            blackout_status[load_node] = False

    # --- Calculate Served Load ---
    # Get current demands and blackout flags as numpy arrays for Numba function
    demands = np.array([graph.nodes[n].get('current_demand_mw', 0) for n in load_nodes], dtype=np.float64)
    blackout_flags = np.array([blackout_status[n] for n in load_nodes], dtype=np.bool_)
    served_load_mw = compute_served_load(demands, blackout_flags, n_loads)

    # --- Overload Check ---
    line_stress_load = defaultdict(float) # Stores calculated load on each line/transmission edge

    # Only proceed if the operating graph has nodes
    if operating_graph_undirected.number_of_nodes() > 0:
        for load_node, is_blacked_out in blackout_status.items():
            if is_blacked_out: continue # Skip blacked-out loads

            load_demand = graph.nodes[load_node].get('current_demand_mw', 0)
            if load_demand <= 0: continue # Skip loads with no demand

            # Find feeders serving this load again
            feeder_sources = [u for u, v, data in graph.edges(data=True)
                              if v == load_node and data.get('in_service', True) and data.get('component') == 'feeder']

            path_found_for_load = False
            for feeder_source_node in feeder_sources:
                # Check if this source is connected to a generator component
                if feeder_source_node in node_to_component and node_to_component[feeder_source_node] in gen_components:
                     comp_idx = node_to_component[feeder_source_node]
                     source_gen = None
                     # Find *any* generator within the same component to use as the path target
                     # (Shortest path calculation assumes power can flow from any gen in the component)
                     for gen in active_generators:
                         if gen in node_to_component and node_to_component[gen] == comp_idx:
                              source_gen = gen
                              break # Found a suitable generator

                     # Ensure both source and target nodes exist in the operating graph before pathfinding
                     if source_gen and operating_graph_undirected.has_node(feeder_source_node) and operating_graph_undirected.has_node(source_gen):
                         try:
                             # Find the shortest path on the *undirected operating graph*
                             # This represents the likely path power would take (ignoring complex flow dynamics)
                             path = shortest_path(operating_graph_undirected, source=feeder_source_node, target=source_gen)

                             # Distribute the load demand along the edges of this path
                             for i in range(len(path) - 1):
                                 u_path, v_path = path[i], path[i + 1]
                                 # Check both directions in the *original directed graph* to assign stress
                                 # We add stress to the edge corresponding to the segment in the path
                                 # This simplification assumes load flows along the shortest path segments
                                 if graph.has_edge(u_path, v_path) and graph[u_path][v_path].get('component') in ['transmission', 'tie_line', 'line']:
                                     line_stress_load[(u_path, v_path)] += load_demand
                                 # Check reverse direction as well, as the undirected path doesn't specify flow direction
                                 elif graph.has_edge(v_path, u_path) and graph[v_path][u_path].get('component') in ['transmission', 'tie_line', 'line']:
                                     line_stress_load[(v_path, u_path)] += load_demand

                             path_found_for_load = True
                             break # Found a path for this load via this feeder, move to next load
                         except nx.NetworkXNoPath:
                             # This might happen if the graph becomes disconnected unexpectedly between checks
                             print(f"Warning: No path found between {feeder_source_node} and {source_gen} despite being in same component.")
                             pass
                         except nx.NodeNotFound:
                             # Should not happen due to checks, but handle defensively
                             print(f"Warning: Node not found during pathfinding ({feeder_source_node} or {source_gen}).")
                             pass

                if path_found_for_load: break # Path found for this load, move to the next load node

    # --- Identify Overloaded Lines ---
    sim_edge_list = list(line_stress_load.keys()) # Edges with calculated stress
    n_sim_edges = len(sim_edge_list)
    overloaded_lines = set() # Set to store (u, v) tuples of overloaded lines

    if n_sim_edges > 0:
        # Ensure edges exist in the main graph before getting capacity data
        valid_edges = [edge for edge in sim_edge_list if graph.has_edge(*edge)]
        if valid_edges:
             n_valid_edges = len(valid_edges)
             # Prepare arrays for Numba function
             stresses = np.array([line_stress_load[edge] for edge in valid_edges], dtype=np.float64)
             # Get capacities, defaulting to infinity if 'capacity_mw' is missing
             capacities = np.array([graph.get_edge_data(u, v).get('capacity_mw', np.inf) for u, v in valid_edges], dtype=np.float64)
             # Pass simple indices [0, 1, ..., n-1] to Numba function
             edge_indices_for_numba = np.arange(n_valid_edges, dtype=np.int64)

             # Call Numba function to get indices of overloaded edges within the 'valid_edges' list
             overloaded_indices = check_overloads(stresses, capacities, edge_indices_for_numba, n_valid_edges)

             # Map the returned indices back to the original edge tuples
             overloaded_lines = {valid_edges[i] for i in overloaded_indices}

    return served_load_mw, blackout_status, overloaded_lines

# Example usage (optional, for testing this module)
if __name__ == '__main__':
    from grid_topology import create_grid_topology
    graph, _, _, _ = create_grid_topology()

    print("Simulating initial state...")
    served, blackouts, overloads = simulate_grid_state(graph)
    print(f"Initial Served Load: {served:.2f} MW")
    print(f"Initial Blackout Status: {blackouts}")
    print(f"Initial Overloaded Lines: {overloads}")

    # Example: Simulate a fault on a critical line
    print("\nSimulating fault on Line 1 (A -> B)...")
    graph_fault = graph.copy()
    if graph_fault.has_edge('Substation A', 'Substation B'):
        graph_fault['Substation A']['Substation B']['in_service'] = False
    # If bi-directional, fault the other way too if needed by model logic
    # if graph_fault.has_edge('Substation B', 'Substation A'):
    #     graph_fault['Substation B']['Substation A']['in_service'] = False

    served, blackouts, overloads = simulate_grid_state(graph_fault)
    print(f"After Fault Served Load: {served:.2f} MW")
    print(f"After Fault Blackout Status: {blackouts}")
    print(f"After Fault Overloaded Lines: {overloads}")
