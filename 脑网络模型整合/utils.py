import torch.nn.functional as F
import torch
import numpy as np
from torch_geometric.data import Data, Batch


def load_net(args):
    if args.v == 'brainGB_GAT':
        from BrainGB.models import GAT, BrainNN
        backbone = GAT(input_dim=load_dim(args.net), num_nodes=90, hidden_dim=128)
        net = BrainNN(gnn=backbone)
    elif args.v == 'brainGB_GCN':
        from BrainGB.models import GCN, BrainNN
        backbone = GCN(input_dim=load_dim(args.net), num_nodes=90, hidden_dim=128)
        net = BrainNN(gnn=backbone)
    elif args.v == 'GAT':
        from models.GAT import GAT
        net = GAT(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'NestedGAT':
        from models.GAT import NestedGAT
        net = NestedGAT(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'GCN':
        from models.GCN import GCN
        net = GCN(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'NestedGCN':
        from models.GCN import NestedGCN
        net = NestedGCN(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'GIN':
        from models.GIN import GIN
        net = GIN(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'NestedGIN':
        from models.GIN import NestedGIN
        net = NestedGIN(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'GlobalAttentionNet':
        from models.global_attention import GlobalAttentionNet
        net = GlobalAttentionNet(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'Graclus':
        from models.graclus import Graclus
        net = Graclus(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'GraphSAGE':
        from models.graph_sage import GraphSAGE
        net = GraphSAGE(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'NestedGraphSAGE':
        from models.graph_sage import NestedGraphSAGE
        net = NestedGraphSAGE(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'SGCN_GCN':
        from models.SGCN import SGCN_GCN
        net = SGCN_GCN(input_dim=load_dim(args.net), hidden=128)
    elif args.v == 'SGCN_GAT':
        from models.SGCN import SGCN_GAT
        net = SGCN_GAT(input_dim=load_dim(args.net), hidden=128)

    else:
        import sys
        sys.exit('指定的模型未设计！')

    """
    网络中加入loss统计
    """
    net.loss_dict = {
        'train_loss': [],
        'eval_loss': [],
        # 'train_loss_multi': {},
        # 'eval_loss_multi': {},
    }

    return net


def load_dim(v):
    if v == 'iRSSN':
        return 43
    elif v == 'R2SN':
        return 22
    elif v.startswith('SwinVIT'):
        if v == 'SwinVIT_D32_P24+rest_fMRI_aal116':
            return 122
        else:
            return 32
    elif v.startswith('CNN'):
        return 32
    elif v.startswith('rest'):
        return 90
    else:
        return 128


def parse_str_or_int(value):
    try:
        # 尝试将输入转换为整数
        return int(value)
    except ValueError:
        # 如果转换失败，则保留为字符串
        return value


def group_to_str(group):
    return '_'.join([str(i) for i in group])


def model_forward_train(args, model, data, device):
    data = data.to(device)
    if args.v.startswith('SGCN'):

        out = model.forward(data)
        loss_ce = F.nll_loss(out, data.y.view(-1))
        out_prob = model(data, True)
        loss_mi = F.nll_loss(out_prob, data.y.view(-1))
        loss_prob = model.loss_probability(data.x, data.edge_index, data.edge_attr, args)
        loss = loss_ce + 2 * loss_prob + loss_mi
    else:
        out = model.forward(data)
        loss = F.nll_loss(out, data.y.view(-1))
    return loss


def model_forward_eval(args, model, data, device):
    data = data.to(device)
    if args.v.startswith('SGCN'):
        out = model(data, True)
    else:
        out = model(data)
    loss_val, loss_val_no_reduction = F.nll_loss(out, data.y.view(-1)), F.nll_loss(out, data.y.view(-1), reduction='none')
    return out, loss_val, loss_val_no_reduction


def preprocess(data, y, threshold, sparsity_method):
    data_list = []
    for i in range(data.shape[0]):
        # 将节点特征 x 转到 CPU，并转置
        x = data[i, ...].cpu().T  # 确保数据在 CPU 上进行操作
        # x_norm = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        # 计算皮尔森相关矩阵
        pc = np.corrcoef(x)
        pc = np.nan_to_num(pc)  # 将 NaN 转换为 0
        pc = abs(pc)  # 取绝对值
        # pc_normal = (pc - np.min(pc)) / (np.max(pc) - np.min(pc))
        # pc_normal = (pc - np.mean(pc)) / np.std(pc)
        np.fill_diagonal(pc, 0)  # 将对角线元素置为 0

        if sparsity_method == 'mask':
            # 找出相关性大于 0.9 的边
            row, col = np.where(pc > threshold)
        elif sparsity_method == 'percent':
            # 计算要保留的最大值个数（至少保留一个）
            num_top = max(1, int(pc.size * threshold))

            # 将数组扁平化后找出前 num_top 个最大值的索引（未排序）
            flat_indices = np.argpartition(pc.flatten(), -num_top)[-num_top:]

            # 转换为二维坐标索引（row, col）
            row, col = np.unravel_index(flat_indices, pc.shape)
        else:
            raise ValueError('sparsity_method属性必须是percent或者mask！')

        # 构建边特征 (edge_attr) 和边连接 (edge_index)
        edge_attr = torch.tensor(pc[row, col], dtype=torch.float)
        edge = torch.tensor(list(zip(row, col)), dtype=torch.long).T

        # 构建图数据对象
        graph_data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge, edge_attr=edge_attr, y=y[i])
        data_list.append(graph_data)
    return Batch.from_data_list(data_list)
