import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import time
import numpy as np
import tqdm
import argparse
from dgl.data import RedditDataset


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def main(args):
    dataset = RedditDataset()
    graph = dataset[0]
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    train_idx = torch.where(train_mask == True)[0]
    valid_idx = torch.where(val_mask == True)[0]

    device = 'cuda'
    train_idx = train_idx.to(device)

    if args.mode == "gpu":
        graph = graph.to(device)  # Pure GPU Sample

    model = SAGE(graph.ndata['feat'].shape[1], 256, dataset.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    sampler = dgl.dataloading.NeighborSampler(
            [10, 25], prefetch_node_feats=['feat'], prefetch_labels=['label'])
    use_uva = args.mode == "uva"
    train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler, device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=use_uva)

    for i in range(args.epochs):
        model.train()
        t0 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        t1 = time.time()
        print(t1 - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGL')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--mode", type=str, default="uva", help="uva, gpu") 
    args = parser.parse_args()
    print(args)
    main(args) 
      
