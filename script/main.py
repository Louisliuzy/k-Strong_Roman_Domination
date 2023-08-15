"""
Main file
Using Gurobi to solve kSRDP
"""

# import
import numpy as np
from kSRDP import kSRDP, Lshaped_kSRDP
from graph_generator import graph_generator


def ip_solver():
    """
    solve using traditional IP method.
    """
    # nodes and attack numbers
    list_nodes = [10]
    list_graphs = [1, 2, 3, 4, 5]
    list_attacks = [2, 3, 4, 5]
    # iteration
    for node in list_nodes:
        for graph in list_graphs:
            txt_name = "graphs/{}_{}.txt"
            adj = np.loadtxt(txt_name.format(node, graph),
                             delimiter=",", usecols=range(node))
            for attack in list_attacks:
                model_name = "{}_{}_{}".format(
                    node, graph, attack
                )
                kSRDP(model_name, adj, attack)


def Lshaped():
    """
    solving using Benders decomposition method
    """
    # nodes and attack numbers
    list_nodes = [10]
    list_graphs = [1, 2, 3, 4, 5]
    list_attacks = [2, 3, 4, 5]
    # iteration
    for node in list_nodes:
        for graph in list_graphs:
            txt_name = "graphs/{}_{}.txt"
            adj = np.loadtxt(txt_name.format(node, graph),
                             delimiter=",", usecols=range(node))
            for attack in list_attacks:
                model_name = "{}_{}_{}".format(
                    node, graph, attack
                )
                Lshaped_kSRDP(
                    name=model_name,
                    adjacency=adj,
                    attack=attack
                )
    return


def graph_generation():
    """
    generating graphs
    """
    for i in range(10, 50, 1):
        graph_generator(
            num_nodes=100,
            num_edges=200,
            iterations=i,
            file_local="graphs/"
        )

    return 0


# main function
def main():
    """main function"""
    # graph_generation()
    # ip_solver()
    Lshaped()


# run
if __name__ == "__main__":
    main()
