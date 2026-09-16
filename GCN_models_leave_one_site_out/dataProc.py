import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data


# /media/shucheng/数程SSD_2T/ADNI/derivatives/epoch20_SwinVIT_D64_P24/data/iDRSN_Cosine_mwp1002_S_0295.csv
# /media/shucheng/数程SSD_2T/ADNI/derivatives/iRSSN/data/iRSSN_mwp1002_S_0295.csv
def cosine_network_by_sparsity(
        node_features: np.ndarray,
        alpha: float = 0.85,
        nonnegative=True,
        method='Cosine',
):
    # # 每个节点特征向量的 L2 范数
    # norms = np.linalg.norm(
    #     node_features,
    #     axis=1,
    #     keepdims=True,
    # )
    # node_features = (
    #     node_features / norms
    # )

    if method == 'PCC':
        S = np.corrcoef(node_features)
        S = np.nan_to_num(S, nan=0.0)
        S = np.abs(S)
        np.fill_diagonal(S, 1.0)

    elif method == 'Cosine':
        # Cosine similarity
        S = cosine_similarity(node_features)
        S = np.clip(S, -1.0, 1.0, )
        if nonnegative:
            S = (S + 1.0) / 2.0
        np.fill_diagonal(S, 1.0)
    else:
        raise ValueError('Method {} not supported'.format(method))
    # 稀疏邻接矩阵
    num_nodes = S.shape[0]

    rows, cols = np.triu_indices(
        num_nodes,
        k=1,
    )

    upper_weights = S[rows, cols]

    total_possible_edges = len(
        upper_weights
    )

    # 根据alpha计算保留边数
    num_edges_to_keep = int(
        round(alpha * total_possible_edges)
    )

    num_edges_to_keep = max(
        1,
        min(
            num_edges_to_keep,
            total_possible_edges,
        ),
    )

    # 选择权重最大的边
    selected_indices = np.argpartition(
        upper_weights,
        -num_edges_to_keep,
    )[-num_edges_to_keep:]

    selected_rows = rows[selected_indices]
    selected_cols = cols[selected_indices]
    selected_weights = upper_weights[selected_indices]

    # 构建双向边
    source_nodes = np.concatenate(
        [
            selected_rows,
            selected_cols,
        ],
        axis=0,
    )

    target_nodes = np.concatenate(
        [
            selected_cols,
            selected_rows,
        ],
        axis=0,
    )

    edge_index = torch.from_numpy(
        np.stack(
            [
                source_nodes,
                target_nodes,
            ],
            axis=0,
        )
    ).long()

    # 双向边权重同步复制
    undirected_edge_weights = np.concatenate(
        [
            selected_weights,
            selected_weights,
        ],
        axis=0,
    )

    edge_weight = torch.from_numpy(
        undirected_edge_weights
    ).float()

    # 每条边只有一个特征
    edge_attr = edge_weight.unsqueeze(-1)

    x = torch.from_numpy(
        node_features
    ).float()

    # # 检查
    # assert edge_index.shape[1] == edge_weight.shape[0]
    # assert edge_index.shape[1] == edge_attr.shape[0]
    #
    # print("x:", x.shape)
    # print("edge_index:", edge_index.shape)
    # print("edge_weight:", edge_weight.shape)
    # print("edge_attr:", edge_attr.shape)

    return x, edge_index, edge_weight, edge_attr
    # data = Data(
    #     # 节点特征：[116, 64]
    #     x=torch.from_numpy(
    #         node_features
    #     ).float(),
    #
    #     # 边索引：[2, 2 × 无向边数]
    #     edge_index=edge_index,
    #
    #     # 一维边权：[num_edges]
    #     # GCNConv 等模型可直接使用
    #     edge_weight=edge_weight,
    #
    #     # 二维边属性：[num_edges, 1]
    #     # GATConv、TransformerConv 等可使用
    #     edge_attr=edge_weight.unsqueeze(1),
    #
    #     # 图分类标签：[1]
    #     y=torch.tensor(
    #         [int(label)],
    #         dtype=torch.long,
    #     ),
    # )


class MedicalDataset(Dataset):
    def __init__(self, dataframe, network='BrainANet_SwinVIT', group=['CN', 'AD'], task='Dementia'):
        """
        dataframe 至少包含：
        Subject_ID, Site, Phase, Group, Label, DataPath

        Label:
            0 -> NC
            1 -> Disease
        """
        dataframe = dataframe[dataframe['Task'] == task]
        # dataframe[dataframe['Group'].isin(group)]
        dataframe = dataframe[
            (~pd.isna(dataframe['Sex'])) & (~pd.isna(dataframe['Age'])) & (
                ~pd.isna(dataframe['Stage'])) & (dataframe['Group'].isin(group))]

        # dataframe = dataframe[~pd.isna('Group')]
        dataframe['Sex'].replace({'M': 0, 'F': 1}, inplace=True)
        self.dict = {
            j: i for i, j in enumerate(group)
        }
        dataframe['Label'] = dataframe['Group'].replace(self.dict, inplace=False)
        self.dataframe = dataframe.reset_index(drop=True)
        self.network = network

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # 替换成你自己的数据读取代码
        node_features, edge_index, edge_weight, edge_attr = cosine_network_by_sparsity(
            pd.read_csv(row[f"Path_{self.network}_node"]).values[:90, :])
        label = self.dict[row['Group']]
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            y=torch.tensor(
                [int(label)],
                dtype=torch.long,
            ),

            site=str(row["Stage"]),
            subject_id=str(row["Subject ID"]),

        )
        # return {
        #     "image": image.float(),
        #     "label": torch.tensor(int(row["Label"]), dtype=torch.long),
        #     "subject_id": str(row["Subject_ID"]),
        #     "site": str(row["Site"]),
        #     "phase": str(row["Phase"]),
        #     "group": str(row["Group"]),
        # }

    def get_sample_weight(self):
        weights = []
        count_nums = np.arange(0, 2).astype(np.int64)
        count = float(self.index.shape[0])
        label = self.to_label(self.dataframe.loc[self.dataframe.tolist(), 'Group'])
        count_class_list = [float(label.count(i)) for i in count_nums]
        for i in label:
            for j in count_nums:
                if i == j:
                    weights.append(count / count_class_list[j])
        imbalanced_ratio = [count_class_list[0] / i_r for i_r in count_class_list]
        return weights, imbalanced_ratio

    def to_label(self, group_list):
        label = []
        for i in group_list:
            cls = self.dict[i]
            label.append(cls)
        return label
