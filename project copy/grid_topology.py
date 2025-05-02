import networkx as nx
import numpy as np

def create_grid_topology():
    """
    Creates the NetworkX graph representing the power grid topology.

    Returns:
        tuple: Contains the graph (G), initial edge states, initial load demands,
               and node positions for plotting.
    """
    G = nx.DiGraph() # Use a directed graph

    # Define node types and add nodes to the graph
    node_types = {
        'Generator': ['Generator 1 (G1)', 'Generator 2 (G2)'],
        'Substation': ['Substation A', 'Substation B', 'Substation C', 'Substation D', 'Substation E', 'Substation F', 'Substation G'],
        'Load Zone': ['Load Zone 1 (Critical)', 'Load Zone 2', 'Load Zone 3'],
    }
    for node_type, nodes in node_types.items():
        for node in nodes:
            G.add_node(node, type=node_type)

    # Define edges with attributes (capacity, initial state, component type)
    # Note: Using DiGraph means defining edges in both directions where applicable (e.g., tie lines)
    edges = [
        # Generators to Substations (High Voltage Transmission)
        ('Generator 1 (G1)', 'Substation A', {'label': 'HV Transmission 1', 'capacity_mw': 500, 'in_service': True, 'critical': True, 'component': 'transmission'}),
        ('Generator 2 (G2)', 'Substation F', {'label': 'HV Transmission 2', 'capacity_mw': 400, 'in_service': True, 'critical': True, 'component': 'transmission'}),

        # Tie Line between Substations A and F (bi-directional)
        ('Substation A', 'Substation F', {'label': 'Line 10 (Tie)', 'capacity_mw': 300, 'in_service': True, 'critical': True, 'component': 'tie_line'}),
        ('Substation F', 'Substation A', {'label': 'Line 10 (Tie)', 'capacity_mw': 300, 'in_service': True, 'critical': True, 'component': 'tie_line'}), # Reverse direction

        # Transmission/Distribution Lines between Substations
        ('Substation A', 'Substation B', {'label': 'Line 1', 'capacity_mw': 150, 'in_service': True, 'critical': True, 'component': 'line'}),
        ('Substation A', 'Substation C', {'label': 'Line 2', 'capacity_mw': 150, 'in_service': True, 'critical': True, 'component': 'line'}),
        ('Substation F', 'Substation C', {'label': 'Line 8', 'capacity_mw': 100, 'in_service': True, 'critical': True, 'component': 'line'}),
        ('Substation F', 'Substation G', {'label': 'Line 9', 'capacity_mw': 100, 'in_service': True, 'critical': True, 'component': 'line'}),
        ('Substation B', 'Substation C', {'label': 'Line 3', 'capacity_mw': 80, 'in_service': True, 'critical': False, 'component': 'line'}),
        ('Substation B', 'Substation D', {'label': 'Line 4', 'capacity_mw': 70, 'in_service': True, 'critical': True, 'component': 'line'}),
        ('Substation C', 'Substation E', {'label': 'Line 5', 'capacity_mw': 90, 'in_service': True, 'critical': True, 'component': 'line'}),
        ('Substation C', 'Substation G', {'label': 'Line 11', 'capacity_mw': 60, 'in_service': True, 'critical': False, 'component': 'line'}),
        ('Substation G', 'Substation E', {'label': 'Line 12', 'capacity_mw': 70, 'in_service': True, 'critical': False, 'component': 'line'}),

        # Bi-directional line between D and E
        ('Substation D', 'Substation E', {'label': 'Line 14', 'capacity_mw': 50, 'in_service': True, 'critical': False, 'component': 'line'}),
        ('Substation E', 'Substation D', {'label': 'Line 14', 'capacity_mw': 50, 'in_service': True, 'critical': False, 'component': 'line'}), # Reverse direction

        # Feeders from Substations to Load Zones (typically uni-directional power flow)
        ('Substation D', 'Load Zone 1 (Critical)', {'label': 'Feeder D-LZ1', 'capacity_mw': 100, 'in_service': True, 'critical': False, 'component': 'feeder'}),
        ('Substation B', 'Load Zone 1 (Critical)', {'label': 'Feeder B-LZ1', 'capacity_mw': 80, 'in_service': True, 'critical': False, 'component': 'feeder'}),
        ('Substation D', 'Load Zone 2', {'label': 'Feeder D-LZ2', 'capacity_mw': 120, 'in_service': True, 'critical': False, 'component': 'feeder'}),
        ('Substation G', 'Load Zone 3', {'label': 'Feeder G-LZ3', 'capacity_mw': 90, 'in_service': True, 'critical': False, 'component': 'feeder'}),
        ('Substation E', 'Load Zone 3', {'label': 'Feeder E-LZ3', 'capacity_mw': 70, 'in_service': True, 'critical': False, 'component': 'feeder'}),
    ]
    G.add_edges_from(edges)

    # Define initial load demands and criticality for Load Zone nodes
    G.nodes['Load Zone 1 (Critical)']['initial_demand_mw'] = 150
    G.nodes['Load Zone 1 (Critical)']['current_demand_mw'] = 150 # Initial current demand matches initial
    G.nodes['Load Zone 1 (Critical)']['critical'] = True

    G.nodes['Load Zone 2']['initial_demand_mw'] = 80
    G.nodes['Load Zone 2']['current_demand_mw'] = 80
    G.nodes['Load Zone 2']['critical'] = False

    G.nodes['Load Zone 3']['initial_demand_mw'] = 120
    G.nodes['Load Zone 3']['current_demand_mw'] = 120
    G.nodes['Load Zone 3']['critical'] = False

    # Store initial states for resetting the environment
    initial_edge_states = {(u, v): data['in_service'] for u, v, data in G.edges(data=True)}
    initial_load_demands = {n: G.nodes[n]['initial_demand_mw']
                            for n, data in G.nodes(data=True) if data['type'] == 'Load Zone'}

    # Define node positions for visualization
    pos = {
        'Generator 1 (G1)': (-3, 7), 'Substation A': (0, 5), 'Substation F': (3, 5),
        'Generator 2 (G2)': (6, 7), 'Substation B': (-2, 3), 'Substation C': (1, 3),
        'Substation G': (4, 3), 'Substation D': (-2, 1), 'Substation E': (3, 1),
        'Load Zone 1 (Critical)': (-3, -1), 'Load Zone 2': (0, -1), 'Load Zone 3': (5, -1),
    }
    # Filter positions to only include nodes actually in the graph
    pos = {node: p for node, p in pos.items() if node in G.nodes()}

    return G, initial_edge_states, initial_load_demands, pos

# Example usage (optional, for testing this module)
if __name__ == '__main__':
    graph, init_edges, init_loads, positions = create_grid_topology()
    print("Graph created successfully.")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Initial Loads: {init_loads}")
    # print(f"Initial Edge States: {init_edges}") # Can be long
    print(f"Node Positions: {positions}")
