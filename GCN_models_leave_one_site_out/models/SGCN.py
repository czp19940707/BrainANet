import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class SGCN_GAT(torch.nn.Module):

    def __init__(self, hidden, hidden_linear=64, rois=90, input_dim=3, num_classes=2, num_layers=2,
                 pooling="concat", **kwargs):
        super(SGCN_GAT, self).__init__()
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.rois = rois
        self.prob_dim = input_dim
        self.conv1 = GATConv(input_dim, hidden, edge_dim=1)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden, edge_dim=1))
        if pooling == "concat":
            gcn_out_dim = rois * num_layers * hidden
        elif pooling == "sum":
            gcn_out_dim = 2 * num_layers * hidden
        self.lin1 = torch.nn.Linear(gcn_out_dim, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))
        self.pooling = pooling

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(self, x, edge_index, edge_weight):
        N, D = x.shape
        x = x.reshape(N // self.rois, self.rois, D)
        x_prob = self.prob  # torch.sigmoid(self.prob) #self.prob
        x_feat_prob = x * x_prob
        x_feat_prob = x_feat_prob.reshape(N, D)
        # print(x_prob)

        conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
        edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
        edge_weight_prob = edge_weight * edge_prob
        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, eps=1e-6):
        _, _, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)

        x_prob = torch.sigmoid(x_prob)

        N, D = x_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num

        N = edge_prob.shape[0]
        all_num = N
        # e_sum_loss = torch.sum(edge_prob)/all_num
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

        # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
        loss_prob = 0.002 * f_sum_loss + 0.1 * e_sum_loss + 0.1 * f_entrp_loss + 0.1 * e_entrp_loss

        return loss_prob

    def forward(self, data, isExplain=False):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        x.requires_grad = True
        self.input = x

        if isExplain:
            x_prob, edge_weight_prob, _, _ = self.cal_probability(x, edge_index, edge_weight)
        else:
            x_prob, edge_weight_prob = x, edge_weight

        x = F.relu(self.conv1(x_prob, edge_index, edge_weight_prob))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight_prob))
            xs += [x]

        x = torch.cat(xs, dim=1)

        if self.pooling == "concat":
            fill_value = x.min().item() - 1
            batch_x, _ = to_dense_batch(x, batch, fill_value)
            B, N, D = batch_x.size()
            z2 = batch_x.view(B, -1)
            x = z2
        elif self.pooling == "sum":
            x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
        # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# class SGCN_GCN(torch.nn.Module):
#     def __init__(self, hidden, hidden_linear=64, rois=90, input_dim=3, num_classes=2, num_layers=2,
#                  pooling="concat", **kwargs):
#         super(SGCN_GCN, self).__init__()
#         self.input = None
#         self.final_conv_acts = None
#         self.final_conv_grads = None
#         self.rois = rois
#         self.prob_dim = input_dim
#         self.conv1 = GCNConv(input_dim, hidden)
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers - 1):
#             self.convs.append(GCNConv(hidden, hidden))
#         self.pooling = pooling
#         if pooling == "concat":
#             gcn_out_dim = rois * num_layers * hidden
#         elif pooling == "sum":
#             gcn_out_dim = 2 * num_layers * hidden
#
#         self.lin1 = torch.nn.Linear(gcn_out_dim, hidden_linear)
#         self.lin2 = Linear(hidden_linear, num_classes)
#
#         self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))
#         self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
#         init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
#         self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
#         init.kaiming_uniform_(self.prob, a=math.sqrt(5))
#         init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))
#
#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#
#         self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))
#         self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
#         init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
#         self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
#         init.kaiming_uniform_(self.prob, a=math.sqrt(5))
#         init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))
#
#     def activations_hook(self, grad):
#         self.final_conv_grads = grad
#
#     def cal_probability(self, x, edge_index, edge_weight):
#         N, D = x.shape
#         x = x.reshape(N // self.rois, self.rois, D)
#         x_prob = self.prob  # torch.sigmoid(self.prob)
#         x_feat_prob = x * x_prob
#         x_feat_prob = x_feat_prob.reshape(N, D)
#
#         conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
#         edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
#         edge_weight_prob = edge_weight * edge_prob
#         return x_feat_prob, edge_weight_prob, x_prob, edge_prob
#
#     def loss_probability(self, x, edge_index, edge_weight, hp, eps=1e-6):
#         _, _, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)
#
#         x_prob = torch.sigmoid(x_prob)
#
#         N, D = x_prob.shape
#         all_num = (N * D)
#         # f_sum_loss = torch.sum(x_prob)/all_num
#         f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N
#         f_entrp_loss = -torch.sum(
#             x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num
#
#         N = edge_prob.shape[0]
#         all_num = N
#         # e_sum_loss = torch.sum(edge_prob)/all_num
#         e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
#         e_entrp_loss = -torch.sum(
#             edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num
#
#         # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
#         # loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss
#         loss_prob = 0.002 * f_sum_loss + 0.1 * e_sum_loss + 0.1 * f_entrp_loss + 0.1 * e_entrp_loss
#
#         return loss_prob
#
#     def forward(self, data, isExplain=False):
#         x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
#         x.requires_grad = True
#         self.input = x
#
#         if isExplain:
#             x_prob, edge_weight_prob, _, _ = self.cal_probability(x, edge_index, edge_weight)
#         else:
#             x_prob, edge_weight_prob = x, edge_weight
#
#         x = F.relu(self.conv1(x_prob, edge_index, edge_weight_prob))
#         xs = [x]
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index, edge_weight_prob))
#             xs += [x]
#
#         x = torch.cat(xs, dim=1)
#
#         if self.pooling == "concat":
#             fill_value = x.min().item() - 1
#             batch_x, _ = to_dense_batch(x, batch, fill_value)
#             B, N, D = batch_x.size()
#             z2 = batch_x.view(B, -1)
#             x = z2
#         elif self.pooling == "sum":
#             x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
#
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#
#         # return F.log_softmax(x, dim=-1)
#         return x
#
#     def __repr__(self):
#         return self.__class__.__name__

class SGCN_GCN(torch.nn.Module):

    def __init__(
            self,
            hidden,
            hidden_linear=64,
            rois=90,
            input_dim=3,
            num_classes=2,
            num_layers=2,
            pooling="concat",
            **kwargs,
    ):
        super().__init__()

        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        self.rois = rois
        self.prob_dim = input_dim
        self.num_layers = num_layers
        self.pooling = pooling

        self.conv1 = GCNConv(
            input_dim,
            hidden,
        )

        self.convs = torch.nn.ModuleList([
            GCNConv(hidden, hidden)
            for _ in range(num_layers - 1)
        ])

        if pooling == "concat":
            gcn_out_dim = (
                rois * num_layers * hidden
            )
        elif pooling == "sum":
            gcn_out_dim = (
                2 * num_layers * hidden
            )
        else:
            raise ValueError(
                f"不支持的pooling方式：{pooling}"
            )

        self.lin1 = torch.nn.Linear(
            gcn_out_dim,
            hidden_linear,
        )

        self.lin2 = torch.nn.Linear(
            hidden_linear,
            num_classes,
        )

        self.prob = torch.nn.Parameter(
            torch.empty(
                self.rois,
                self.prob_dim,
            )
        )

        self.prob_bias = torch.nn.Parameter(
            torch.empty(
                self.prob_dim * 2,
                1,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        init.kaiming_uniform_(
            self.prob,
            a=math.sqrt(5),
        )

        init.kaiming_uniform_(
            self.prob_bias,
            a=math.sqrt(5),
        )

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(
            self,
            x,
            edge_index,
            edge_weight,
    ):
        num_nodes, feature_dim = x.shape

        if feature_dim != self.prob_dim:
            raise ValueError(
                f"x的特征维度为{feature_dim}，"
                f"prob_dim为{self.prob_dim}。"
            )

        if num_nodes % self.rois != 0:
            raise ValueError(
                f"总节点数{num_nodes}不能被"
                f"rois={self.rois}整除。"
            )

        edge_weight = edge_weight.reshape(-1)

        if edge_index.size(1) != edge_weight.numel():
            raise ValueError(
                f"边数为{edge_index.size(1)}，"
                f"边权数为{edge_weight.numel()}。"
            )

        batch_size = num_nodes // self.rois

        x_3d = x.reshape(
            batch_size,
            self.rois,
            feature_dim,
        )

        node_prob = torch.sigmoid(
            self.prob
        )

        x_prob = (
            x_3d * node_prob.unsqueeze(0)
        ).reshape(
            num_nodes,
            feature_dim,
        )

        source = edge_index[0]
        target = edge_index[1]

        # 对称边特征，适合无向脑网络
        edge_features = torch.cat(
            [
                x_prob[source] + x_prob[target],
                torch.abs(
                    x_prob[source] -
                    x_prob[target]
                ),
            ],
            dim=-1,
        )

        edge_prob = torch.sigmoid(
            edge_features.matmul(
                self.prob_bias
            )
        ).reshape(-1)

        edge_weight_prob = (
            edge_weight * edge_prob
        )

        return (
            x_prob,
            edge_weight_prob,
            node_prob,
            edge_prob,
        )

    def forward(
            self,
            data,
            use_probability=True,
            isExplain=False,
            return_features=False,
    ):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = data.edge_weight.reshape(-1)

        if edge_index.size(1) != edge_weight.numel():
            raise ValueError(
                f"edge_index包含"
                f"{edge_index.size(1)}条边，"
                f"edge_weight包含"
                f"{edge_weight.numel()}个权重。"
            )

        node_counts = torch.bincount(batch)

        if not torch.all(
            node_counts == self.rois
        ):
            raise ValueError(
                "每个图必须具有固定脑区数，"
                f"当前为{node_counts.tolist()}。"
            )

        if isExplain:
            x = (
                x.detach()
                .clone()
                .requires_grad_(True)
            )

        self.input = x

        if use_probability:
            x, edge_weight_used, _, _ = (
                self.cal_probability(
                    x,
                    edge_index,
                    edge_weight,
                )
            )
        else:
            edge_weight_used = edge_weight

        x = F.relu(
            self.conv1(
                x,
                edge_index,
                edge_weight_used,
            )
        )

        xs = [x]

        for conv in self.convs:
            x = F.relu(
                conv(
                    x,
                    edge_index,
                    edge_weight_used,
                )
            )
            xs.append(x)

        x = torch.cat(
            xs,
            dim=1,
        )

        if isExplain:
            self.final_conv_acts = x
            x.register_hook(
                self.activations_hook
            )

        if self.pooling == "concat":
            batch_x, _ = to_dense_batch(
                x,
                batch,
                fill_value=0.0,
                max_num_nodes=self.rois,
            )

            x = batch_x.reshape(
                batch_x.size(0),
                -1,
            )

        elif self.pooling == "sum":
            x = torch.cat(
                [
                    gmp(x, batch),
                    gap(x, batch),
                ],
                dim=1,
            )

        hidden_features = F.relu(
            self.lin1(x)
        )

        classifier_features = F.dropout(
            hidden_features,
            p=0.5,
            training=self.training,
        )

        logits = self.lin2(
            classifier_features
        )

        if return_features:
            return logits, hidden_features

        return logits