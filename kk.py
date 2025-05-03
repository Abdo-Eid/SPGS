import networkx as nx
from project.conf import *
import matplotlib.pyplot as plt

def define_grid_topology():
    """Defines the grid network structure."""
    G = nx.Graph()

    # Add Nodes: Generators, Substations, Load Zones
    G.add_node("G1", type=GENERATOR)
    G.add_node("G2", type=GENERATOR)

    G.add_node("S1", type=SUBSTATION)
    G.add_node("S2", type=SUBSTATION)
    G.add_node("S3", type=SUBSTATION)
    G.add_node("S4", type=SUBSTATION)

    G.add_node("L1", type=LOAD_ZONE, critical=False)
    G.add_node("L2", type=LOAD_ZONE, critical=True) # Critical Load 1
    G.add_node("L3", type=LOAD_ZONE, critical=False)
    G.add_node("L4", type=LOAD_ZONE, critical=True) # Critical Load 2
    G.add_node("L5", type=LOAD_ZONE, critical=False)

    # Add Connections (Edges) with initial status and controllability
    # Define edges as (node1, node2, {'status': initial_status, 'controllable': bool})
    edges = [
        ("G1", "S1", {'controllable': False}),
        ("G2", "S2", {'controllable': False}),

        ("S1", "S2", {'controllable': True}),
        ("S1", "S3", {'controllable': True}),
        ("S2", "S4", {'controllable': True}),
        ("S3", "S4", {'controllable': True}), # Redundant path
        ("S1", "S4", {'controllable': True}), # Another path

        ("S3", "L1", {'controllable': False}),
        ("S3", "L2", {'controllable': False}),
        ("S4", "L3", {'controllable': False}),
        ("S4", "L4", {'controllable': False}),
        ("S2", "L5", {'controllable': False}),
        ("S3", "L5", {'controllable': False}), # L5 has dual supply potential
    ]

    # add all edjes to be IN_SERVICE so we don't need to add it manually since they all will be IN_SERVICE
    for u, v, attrs in edges:
        attrs['status'] = IN_SERVICE
        G.add_edge(u, v, **attrs)

    return G

G = define_grid_topology()
nx.draw(G, pos=nx.spring_layout(G, seed= 41), with_labels=True)
plt.show()