"""
Created on Thur May 13 2021

@author: Jaime Enriquez Ballesteros, @ebjaime
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_barabasi_albert_graph(num_nodes, num_edges, to_file, random_seed=42):
    graph = nx.generators.random_graphs.barabasi_albert_graph(num_nodes, num_edges, random_seed)
    if to_file is not None:
        nx.write_edgelist(graph, to_file, data=False, delimiter=",")
    return graph

def generate_erdos_renyi_graph(num_nodes, prob, to_file, random_seed=42):
    graph = nx.generators.random_graphs.erdos_renyi_graph(num_nodes, prob, random_seed)
    if to_file is not None:
        nx.write_edgelist(graph, to_file, data=False, delimiter=",")
    return graph

def draw_graph(graph):
    nx.draw(graph, with_labels=True, font_color="white", node_size=700)

def plot_coords(x, y, text=None):
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(str(i+1) if text is None else text[i], (x[i]+.01, y[i]+.01)) #(x[i]-.02, y[i]-0.08) for edges

def random_coords(size):
    coords_x = np.random.rand(size) * 2
    coords_y = np.random.rand(size) * 2
    return coords_x, coords_y

def trans_e_example():
    data = np.array([[1, 2], [4, 1], [2.5, 0.25]])
    origin = np.array([[0, 0, 1], [0, 0, 2]])
    plt.quiver(*origin, data[:, 0], data[:, 1], color=['black', 'black', 'red'], angles='xy', scale_units='xy', scale=1)

    # Error line
    plt.plot([3.5,4], [2.25,1], color="black", linestyle="dotted")
    plt.xlim(0, 5)
    plt.ylim(0, 3)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.show()

def trans_h_example():
    data = np.array([[6,-9/4],[4, 8], [12, 5] ])
    origin = np.array([[4, 0, 0] , [4, 0, 0]])
    plt.quiver(*origin, data[:, 0], data[:, 1], color=['red', 'black', 'black'], angles='xy', scale_units='xy', scale=1)
    # New plane drawing
    plt.plot([0,5.5,20], [0,7, 7.25], color="black", linestyle="dashed")
    plt.plot([0,14.5,20], [0,.25, 7.25], color="black", linestyle="dashed")
    # Parallel lines
    plt.plot([4,4],[8,4], color="blue", linestyle="dotted")
    plt.plot([12,12],[5,1], color="blue", linestyle="dotted")
    # Error line
    plt.plot([10,12],[4-9/4,1], color="black", linestyle="dotted")
    # Color
    plt.fill_between([0,5.5,14.5,20],[0,7, 7.1551, 7.25], [0,.0948,.25, 7.25], zorder=-1, color="black", alpha=0.2)
    plt.xlim(0, 21)
    plt.ylim(0, 10)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.show()

def trans_r_example(relation_space=True):
    if relation_space:
        data = np.array([[2, 3], [5, 1]])
        origin = np.array([[0, 0], [0, 0]])
        plt.quiver(*origin, data[:, 0], data[:, 1], color=['black', 'black'], angles='xy', scale_units='xy', scale=1)
        plt.xlim(0, 7)
        plt.ylim(0, 5)

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.show()
    else:
        data = np.array([[2.5, 0.25]])
        origin = np.array([[1], [2]])
        plt.quiver(*origin, data[:, 0], data[:, 1], color=['red'], angles='xy', scale_units='xy',scale=1)
        # Error line
        plt.plot([3.5, 4], [2.25, 1], color="black", linestyle="dotted")
        plt.xlim(0, 5)
        plt.ylim(0, 5)

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.show()


if __name__ == "__main__":
    NUM_NODES = 9
    try:
        graph = nx.read_edgelist("random_graph.csv", delimiter=",", data=False, nodetype=int)
    except Exception:
        graph = generate_erdos_renyi_graph(NUM_NODES, 0.3, "random_graph.csv")
    draw_graph(graph)

    # coords_nodes_x, coords_nodes_y = random_coords(NUM_NODES)
    # coords_edges_x, coords_edges_y = random_coords(len(graph.edges))
    #
    # plot_coords(coords_nodes_x, coords_nodes_y)
    # plot_coords(coords_edges_x, coords_edges_y, ["e"+str(edge[0])+str(edge[1]) for edge in graph.edges])
    # plot_coords([coords_nodes_x[0]], [coords_nodes_y[0]], ["G"])

    # trans_e_example()
    # trans_h_example()
    # trans_r_example(False)
