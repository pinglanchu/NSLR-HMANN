import time
import dgl
import numpy as np
import torch, gc
from scipy import sparse
import dgl.function as fn
from sklearn.metrics import roc_auc_score, f1_score
import itertools
from Hypergraph import construct_hypergraph
from attention import *
import networkx as nx
from SAGE import *
from GAT import GraphGAT
from HGNN import *
# from HGNNPlus import *
# import dhg
# from dhg.structure.hypergraphs import Hypergraph
# from dhg.nn.convs.hypergraphs import hnhn_conv, hypergcn_conv
# from dhg.nn.convs.graphs import gcn_conv, gin_conv


class Dotpredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


def get_pos_neg_graph(g):
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)  # shuffle
    test_size = int(len(eids) * 0.3)
    # train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sparse.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())  # default replace=True
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    return dgl.add_self_loop(train_pos_g), dgl.add_self_loop(train_neg_g), \
           dgl.add_self_loop(test_pos_g), dgl.add_self_loop(test_neg_g), eids[:test_size]


def load_data(net_name):
    data_dir = '../Data/'
    adjacency = np.load(data_dir + net_name)
    graph = nx.from_numpy_matrix(adjacency)
    n = adjacency.shape[0]
    subject_g = dgl.from_networkx(graph)
    train_pos_g, train_neg_g, test_pos_g, test_neg_g, test_link = get_pos_neg_graph(subject_g)
    train_subject_g = dgl.remove_edges(subject_g, test_link)
    train_subject_g = dgl.add_self_loop(train_subject_g)
    train_network = train_subject_g.to_networkx()
    H_graph, H = construct_hypergraph(nx.to_numpy_array(train_network), is_binary=False)  # adjacency_matrix
    train_g = dgl.from_networkx(H_graph)
    train_g = dgl.add_self_loop(train_g)
    train_g.ndata['feature'] = nn.Embedding(n, 64).weight
    print('dataset loaded')
    return H, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g


# def dhg_construct_hg(H):
#     num_v = H.shape[0]
#     H = H + np.eye(num_v)
#     row, col = np.nonzero(H)
#     hyperedges = dict()
#     for node, edge in zip(row, col):
#         if edge in hyperedges:
#             hyperedges[edge].append(node)
#         else:
#             hyperedges[edge] = [node]
#     for key in hyperedges.keys():
#         hyperedges[key].append(key)
#
#     e_list = list(hyperedges.values())
#     hypergraph = dhg.structure.Hypergraph(num_v, e_list)
#     return hypergraph


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_metrics(pos_score, neg_score):  # scores.astype('int64')
    scores = torch.cat([pos_score.clamp(0, 1), neg_score.clamp(0, 1)]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, np.round(scores)), f1_score(labels, np.round(scores), average='macro')


# 'Bio-CE-GT.npy', 'Celegans.npy', 'Ecoli.npy', 'Facebook.npy',
# 'Jazz.npy', 'Metabolic.npy', 'PB.npy', 'Soc-Wiki-Vote.npy'
auc_arr = np.zeros((5000, 8))
f1_arr = np.zeros((5000, 8))
col = 0
for net_name in ['Bio-CE-GT.npy', 'Celegans.npy', 'Ecoli.npy', 'Facebook.npy',
                 'Jazz.npy', 'Metabolic.npy', 'PB.npy', 'Soc-Wiki-Vote.npy']:
    print(net_name)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    H, g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = load_data(net_name)

    # GCN & GIN
    # graph = g.to_networkx()
    # e_list = list(graph.edges)
    # num_v = graph.number_of_nodes()
    # graph = dhg.structure.Graph(num_v, e_list)

    # HGNN
    # G = generate_G_from_H(H)
    # G = torch.from_numpy(G).float().to(device)

    # HNHN(可以删掉dhg_construct_hg中添加两个中心节点的两行代码) & hypergcn_conv
    # hypergraph = dhg_construct_hg(H).to(device)

    H = torch.from_numpy(H).float().to(device)

    g = g.to(device)

    # loss_record = np.zeros((10000, 10))
    AUC_list = []
    f1_list = []

    model = Multi_view_Attention(in_size=64, hidden_size=32, out_size=32, num_heads=[8, 2], dropout=0.5).to(device)
    # model = gcn_conv.GCNConv(in_channels=64, out_channels=32).to(device)
    # model = gin_conv.GINConv(nn.Linear(64, 32)).to(device)
    # model = GraphSAGE(in_size=64, out_size=32).to(device)
    # model = GraphGAT(in_size=64, out_size=32).to(device)
    # model = HGNN(in_ch=64, n_class=32, n_hid=32).to(device)
    # model = hnhn_conv.HNHNConv(in_channels=64, out_channels=32).to(device)
    # model = hypergcn_conv.HyperGCNConv(in_channels=64, out_channels=32).to(device)

    # pred = Dotpredictor().to(device)
    pred = MLPPredictor(h_feats=32).to(device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

    # start_time = time.time()
    loss_list = []
    for e in range(5000):
        # forward
        gc.collect()
        torch.cuda.empty_cache()

        h = model(g, g.ndata['feature'].to(device), H)  # Ours
        # h = model(g, g.ndata['feature'].to(device))  # GraphSAGE and GAT
        # h = model(g.ndata['feature'].to(device), G.to(device))  # HGNN
        # h = model(g.ndata['feature'].to(device), hypergraph)  # HNHN, hypergcn
        # h = model(g.ndata['feature'].to(device), graph.to(device))  # GCN, GIN

        pos_score = pred(train_pos_g.to(device), h)
        neg_score = pred(train_neg_g.to(device), h)
        loss = compute_loss(pos_score.cpu(), neg_score.cpu())
        # loss_list.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # loss_record[:, repeat_num] = loss_list

        # if e % 10 == 0:
        #     print('In epoch {}, loss: {}'.format(e, loss))

    # end_time = time.time()
    # print('consume time:', end_time - start_time)

        with torch.no_grad():
            pos_score = pred(test_pos_g.to(device), h)
            neg_score = pred(test_neg_g.to(device), h)

            metrics = compute_metrics(pos_score.cpu(), neg_score.cpu())
            AUC_list.append(metrics[0])
            f1_list.append(metrics[1])
    print(metrics[0], metrics[1])
    # auc_arr[:, col] = AUC_list
    # f1_arr[:, col] = f1_list
    col += 1
    # np.save('./loss_matrix/%s' % net_name, loss_record)
# np.save('./loss_matrix/AUC.npy', auc_arr)
# np.save('./loss_matrix/f1.npy', f1_arr)

