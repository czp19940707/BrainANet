import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
import pdb
from torch_geometric.utils import to_dense_batch


class NestedGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden, use_z=False, use_rd=False, num_layers=2, num_classes=2, hidden_linear=64):
        super(NestedGAT, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        if self.use_z or self.use_rd:
            input_dim += 8

        self.conv1 = GATConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # node label embedding
        z_emb = 0
        if self.use_z and 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        if self.use_rd or self.use_z:
            x = torch.cat([z_emb, x], -1)

        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        # x = global_mean_pool(x, data.subgraph_to_graph)

        fill_value = x.min().item() - 1
        batch_x, _ = to_dense_batch(x, data.subgraph_to_graph, fill_value)
        B, N, D = batch_x.size()
        z2 = batch_x.view(B, -1)
        x = z2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)  # ad: 0.5
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GAT(torch.nn.Module):

    def __init__(self, input_dim, hidden, *args, num_classes=2, num_layers=2, hidden_linear=64, **kwargs):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden, edge_dim=1)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden, edge_dim=1))
        self.lin1 = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            xs += [x]
        # x = global_mean_pool(torch.cat(xs, dim=1), batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)

        x = torch.cat(xs, dim=1)

        fill_value = x.min().item() - 1
        batch_x, _ = to_dense_batch(x, batch, fill_value)
        B, N, D = batch_x.size()
        z2 = batch_x.view(B, -1)
        x = z2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
