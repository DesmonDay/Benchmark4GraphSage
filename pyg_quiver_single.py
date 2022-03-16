import os.path as osp
import argparse
import quiver
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)


def main(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True)  # Quiver
    csr_topo = quiver.CSRTopo(data.edge_index)
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10],
                                             device=0, mode=args.mode.upper())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE(dataset.num_features, 256, dataset.num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = data.x.to(device)
    y = data.y.squeeze().to(device)

    for i in range(args.epochs):
        model.train()
        t0 = time.time()
        for seeds in train_loader:  # Quiver
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)  # Quiver
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
        t1 = time.time()
        print(t1 - t0)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Pyg') 
   parser.add_argument("--epochs", type=int, default=20)
   parser.add_argument("--batch_size", type=int, default=1024)
   parser.add_argument("--mode", type=str, default="uva", help="uva, gpu")
   args = parser.parse_args()
   print(args)
   main(args)
