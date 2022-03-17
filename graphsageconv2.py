import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.incubate import graph_send_recv

import pgl
from pgl.nn import functional as GF


class GraphSageConv2(nn.Layer):
    def __init__(self, input_size, hidden_size, normalize=False, pool_type="sum"):
        super(GraphSageConv2, self).__init__()
        self.pool_type = pool_type
        self.normalize = normalize
        self.neigh_linear = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.self_linear = nn.Linear(input_size, hidden_size)

    def forward(self, graph, x, act=None):
        if isinstance(x, paddle.Tensor):
            x = (x, x)

        src, dst = graph
        output = graph_send_recv(x[0], src, dst, pool_type=self.pool_type,
                                  out_size=x[1].shape[0])
        output = self.neigh_linear(output)
        output += self.self_linear(x[1])
        if self.normalize:
            output = F.normalize(output, axis=1)
        return output
