import matplotlib.pyplot as plt
import networkx as nx


def plot_routing_graph(G, title: str = "Routing Graph"):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=300)
    plt.title(title)
    plt.tight_layout()
