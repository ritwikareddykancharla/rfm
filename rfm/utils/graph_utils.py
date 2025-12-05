import networkx as nx
import torch


def generate_random_routing_graph(n_nodes: int = 50, p: float = 0.1):
    """Generate a simple random directed graph for toy routing.

    Returns:
        G: networkx.DiGraph
    """
    G = nx.gnp_random_graph(n_nodes, p, directed=True)
    G = nx.DiGraph((u, v, {"cost": 1.0}) for u, v in G.edges())
    return G
