"""
Random Graph Generation (RGG).
"""

import networkx as nx
import matplotlib.pyplot as plt


def graph_generator(num_nodes, num_edges, iterations, file_local):
    """
    num_nodes: the number of vertices; num_edges: the number of edges;
    iterations: number of iterations
    """
    g = nx.dense_gnm_random_graph(num_nodes, num_edges)
    options = {
        'node_color': 'grey',
        'node_shape': 'o',
        'node_size': 100,
        'width': 1,
        'font_size': 6
    }
    # draw and save
    nx.draw_networkx(g, **options)
    png_name = file_local + "{}_{}.png"

    plt.savefig(png_name.format(num_nodes, iterations), dpi=300)
    adjacency = nx.to_numpy_matrix(g)
    # write to file
    txt_name = file_local + "{}_{}.txt"
    txt_file = open(txt_name.format(num_nodes, iterations), "w+")
    for i in range(0, adjacency.shape[0]):
        for j in range(0, adjacency.shape[1]):
            if i == j:
                adjacency[i, j] = 1
                txt_file.write(str(adjacency[i, j]))
            else:
                txt_file.write(str(adjacency[i, j]))
            if j != adjacency.shape[1] - 1:
                txt_file.write(", ")
        txt_file.write("\n")
    txt_file.close()
    plt.clf()
    return
