import numpy as np
import networkx as nx


def construct_hypergraph(adjacency, is_binary):
    # data_dir = '../Data/'
    # adjacency = np.load(data_dir + net_name)
    n = adjacency.shape[0]

    alpha = 1
    I = np.eye(n)
    Z = alpha * np.linalg.inv(alpha * np.dot(adjacency.T, adjacency) + I).dot(adjacency.T).dot(adjacency)
    # print(Z)
    # facebook:0.015, Bio-CE-GT:
    row, col = np.where(Z > 0.25)  # 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
    H = np.zeros((n, n))

    # binary incidence matrix
    if is_binary:
        H[row, col] = 1

    # probalitical incidence matrix
    else:
        H[row, col] = Z[row, col]

    W_temp = np.dot(H.T, Z)
    W_norm = np.linalg.norm(W_temp, axis=1)
    W = np.diag(W_norm / np.sum(H, axis=0))
    # # W = np.eye(n)
    DV = np.diag(np.sum(np.dot(H, W), axis=1))
    #
    hypergraph_adj = np.dot(H, W).dot(H.T) - DV
    graph = nx.from_numpy_matrix(hypergraph_adj)
    # graph = nx.from_numpy_matrix(adjacency)
    return graph, H + adjacency


# 绘制不同节点为中心的超边
# hyperedges = [edges for edges in zip(row, col)]
# centernode_slavernodes = dict()
# for key, value in hyperedges:
#     if key not in centernode_slavernodes:
#         centernode_slavernodes[key] = [value]
#     else:
#         centernode_slavernodes[key].append(value)

# graph = nx.from_numpy_matrix(adjacency)
# pos = nx.spring_layout(graph)

# for i in range(10):
#     nx.draw_networkx_nodes(graph, pos=pos, nodelist=centernode_slavernodes[i], node_color='red', node_size=10)
#     nx.draw_networkx_nodes(graph, pos=pos, nodelist=[i], node_color='yellow', node_size=10)
#     nx.draw_networkx_edges(graph, pos=pos)
#
#     plt.savefig('hyperedge_%d.png' % i, dpi=300)
#     plt.show()
