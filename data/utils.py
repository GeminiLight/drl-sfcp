import os
import json
import numpy as np
import networkx as nx


def read_json(fpath):
    with open(fpath, 'r') as f:
        attrs_dict = json.load(f)
    return attrs_dict

def write_json(dict_data, fpath):
    with open(fpath, 'w+') as f:
        json.dump(dict_data, f)

def path_to_edges(path):
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def draw_graph(G, width=0.05, show=True, save_path=None):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8), dpi=200)
    size = 10
    edge_colors = None
    options = {
        'pos': pos, 
        "node_color": 'blue',
        "node_size": size,
        "line_color": "grey",
        "linewidths": 0,
        "width": width,
        'with_label': True, 
        "cmap": plt.cm.brg,
        'edge_color': edge_colors,
        'edge_cmap': plt.cm.Blues, 
        'alpha': 0.5, 
    }
    nx.draw(G, **options)
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
