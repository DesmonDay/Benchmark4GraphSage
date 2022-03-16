import time
import argparse

import numpy as np
import paddle
from paddle.fluid import core
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
import pgl
import pgl.graph_kernel as graph_kernel


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

        self.num_layers = 2
        self.convs = nn.LayerList()
        self.convs.append(pgl.nn.GraphSageConv2(input_size, hidden_size))
        self.convs.append(pgl.nn.GraphSageConv2(hidden_size, num_classes))

    def forward(self, graph_list, feat):
        for i, (graph, size) in enumerate(graph_list):
            feat_target = feat[:size]
            feat = self.convs[i](graph, (feat, feat_target))
            if i != self.num_layers - 1:
                feat = F.relu(feat)
                feat = F.dropout(feat, p=0.5)
        return feat


def load_dataset():
    edge_index = np.load("reddit/edge_index.npy")
    feature = np.load("reddit/feature.npy")
    y = np.load("reddit/y.npy")
    train_index = np.load("reddit/train_idx.npy")
    val_index = np.load("reddit/val_idx.npy")
    test_index = np.load("reddit/test_idx.npy")

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


def get_sample_graph_list(row, colptr, nodes, sample_sizes):
    graph_list = []
    for size in sample_sizes:
        neighbors, neighbor_counts = core.ops.graph_sample_neighbors(
            row, colptr, nodes, "sample_size", size)
        edge_src, edge_dst, sample_index = core.ops.graph_reindex(
            neighbors, neighbor_counts, nodes)
        graph = pgl.Graph(num_nodes=len(sample_index),
                          edges=paddle.concat([edge_src.reshape([-1, 1]),
                                              edge_dst.reshape([-1, 1])],
                                              axis=-1))
        graph_list.append((graph, nodes.shape[0]))
        nodes = sample_index
    return graph_list[::-1], nodes


def main(args):
    graph, train_index, val_index, test_index, \
        train_label, val_label, test_label, feature = load_dataset()
    row, colptr = get_basic_graph_sample_neighbors_info(graph, args.mode)
    train_ds = ShardedDataset(train_index, train_label)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    model = GraphSage(feature.shape[1], 41, 2, 256)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    feature = paddle.to_tensor(feature)

    for i in range(args.epochs):
        model.train()
        t0 = time.time()
        for node_index, node_label in train_loader:
            graph_list, n_id = get_sample_graph_list(row, colptr, node_index, [25, 10])
            pred = model(graph_list, feature[n_id])
            loss = criterion(pred, node_label)
            loss.backward()
            optim.step()
            optim.clear_grad()
        t1 = time.time()
        print(t1 - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PGL')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--mode", type=str, default="uva", help="uva, gpu")
    args = parser.parse_args()
    print(args)
    main(args)
