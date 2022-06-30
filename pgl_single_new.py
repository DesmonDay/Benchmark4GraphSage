import time
import argparse
import logging

import numpy as np
import paddle
from paddle.fluid import core
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
import pgl
import pgl.graph_kernel as graph_kernel
from graphsageconv2 import GraphSageConv2

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class ShardedDataset(Dataset):
    def __init__(self, data_index, data_label):
        self.data_index = data_index
        self.data_label = data_label

    def __getitem__(self, idx):
        return self.data_index[idx], self.data_label[idx]

    def __len__(self):
        return len(self.data_index)


class GraphSage(nn.Layer):
    def __init__(self, input_size, num_classes, num_layers=1,
                 hidden_size=64, dropout=0.5, **kwargs):
        super(GraphSage, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.LayerList()
        self.convs.append(GraphSageConv2(input_size, hidden_size))
        self.convs.append(GraphSageConv2(hidden_size, num_classes))
        # self.convs.append(GraphSageConv2(input_size, num_classes))

    def forward(self, graph_list, feat):
        for i, (edge_src, edge_dst, size) in enumerate(graph_list):
            feat_target = feat[:size]
            feat = self.convs[i]((edge_src, edge_dst), (feat, feat_target))
            if i != self.num_layers - 1:
                feat = F.relu(feat)
                feat = F.dropout(feat, 0.5, training=self.training)
        return feat


def load_dataset():
    edge_index = np.load("../reddit/edge_index.npy")
    feature = np.load("../reddit/feature.npy")
    y = np.load("../reddit/y.npy")
    train_index = np.load("../reddit/train_idx.npy")
    val_index = np.load("../reddit/val_idx.npy")
    test_index = np.load("../reddit/test_idx.npy")

    train_label = y[train_index]
    val_label = y[val_index]
    test_label = y[test_index]

    graph = pgl.Graph(num_nodes=232965,
                      edges=edge_index.T)

    return graph, train_index, val_index, test_index, \
           train_label, val_label, test_label, feature

def get_basic_graph_sample_neighbors_info(graph, mode="uva"):
    u = graph.edges[:, 0]
    v = graph.edges[:, 1]
    _, row, _, _, colptr = graph_kernel.build_index(v, u, graph.num_nodes)
    row = row.astype(np.int64)

    if mode == "uva":
        row = core.to_uva_tensor(row)
    else:
        row = paddle.to_tensor(row)

    colptr = paddle.to_tensor(colptr, dtype="int64")
    return row, colptr


class NeighborSampler(object):
    def __init__(self, graph, sample_size, mode="uva"):
        self.row, self.colptr = get_basic_graph_sample_neighbors_info(graph, mode=mode)
        self.graph = graph
        self.sample_size = sample_size
        self.key_buffer = paddle.full([self.graph.num_nodes], -1, dtype="int32")
        self.value_buffer = paddle.full([self.graph.num_nodes], -1, dtype="int32")
        self.mode = mode

        if mode == "gpu":
            self.eid_perm = paddle.arange(0, self.graph.num_edges, dtype="int64") #paddle.to_tensor(self.row.numpy())
        else:
            self.eid_perm = None

    def sample(self, nodes):
        graph_list = []
        for size in self.sample_size:
            if self.mode == "gpu" and nodes.shape[0] > 1024:
                neighbors, neighbor_counts = paddle.incubate.graph_sample_neighbors(
                    self.row, self.colptr, nodes, perm_buffer=self.eid_perm, 
                    sample_size=size, flag_perm_buffer=True)
            else:
                neighbors, neighbor_counts = paddle.incubate.graph_sample_neighbors(
                    self.row, self.colptr, nodes, sample_size=size)

            edge_src, edge_dst, sample_index = paddle.incubate.graph_reindex(
                nodes, neighbors, neighbor_counts, self.key_buffer, 
                self.value_buffer, True)

            graph_list.append((edge_src, edge_dst, nodes.shape[0]))
            nodes = sample_index
        return graph_list[::-1], nodes
    

def get_sample_graph_list(row, colptr, nodes, sample_sizes, key_buffer, value_buffer):
    graph_list = []
    #timer = []
    sample_t = []
    reindex_t = []
    
    for size in sample_sizes:
        t1 = time.perf_counter()
        neighbors, neighbor_counts = core.ops.graph_sample_neighbors(
            row, colptr, nodes, "sample_size", size)
        t2 = time.perf_counter()

        edge_src, edge_dst, sample_index, key_buffer, value_buffer = core.ops.graph_reindex(
            nodes, neighbors, neighbor_counts, key_buffer, value_buffer,
            "flag_buffer_hashtable", True)

        t3 = time.perf_counter()
        graph_list.append((edge_src, edge_dst, nodes.shape[0]))
        nodes = sample_index
        sample_t.append(t2 - t1)
        reindex_t.append(t3 - t2)
        #timer.append((t2 - t1, t3 - t2, len(nodes), size))
    return graph_list[::-1], nodes, np.sum(sample_t), np.sum(reindex_t)


def main(args):
    graph, train_index, val_index, test_index, \
        train_label, val_label, test_label, feature = load_dataset()
    train_ds = ShardedDataset(train_index, train_label)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    model = GraphSage(feature.shape[1], 41, 2, 256)
    criterion = paddle.nn.loss.CrossEntropyLos
    optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    feature = paddle.to_tensor(feature)

    ns = NeighborSampler(graph, [25, 10], mode=args.mode)

    total = []

    logging.info("start")
    for i in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        for node_index, node_label in train_loader:
            graph_list, n_id = ns.sample(node_index)
            
            pred = model(graph_list, feature[n_id])
            loss = criterion(pred, node_label)
            loss.backward()
            optim.step()
            optim.clear_grad()
        t1 = time.perf_counter()
        total.append(t1 - t0)
        logging.info(t1 - t0) #, np.sum(total_sample_t), np.sum(total_reindex_t))
    logging.info("pgl, batch_size:%d, mode:%s, %f" % (args.batch_size, args.mode, np.mean(total[4:])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PGL')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--mode", type=str, default="uva", help="uva, gpu")
    args = parser.parse_args()
    main(args)

