import os

os.environ["LOKY_MAX_CPU_COUNT"] = "16"
import os
import pandas as pd
import torch
from dataProc import cosine_network_by_sparsity
from torch_geometric.data import Data
from models.GAT import GAT
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
from train import load_dim


def plot_tsne_two_groups(
        group1_features,
        group2_features,
        group1_name="Group 1",
        group2_name="Group 2",
        standardize=True,
        perplexity=10,
        random_state=42,
        save_path=None,
        dpi=300,
        show_figure=False,
):
    """
    将两组64维特征统一降维到二维并绘制t-SNE散点图。

    Parameters
    ----------
    group1_features : list/ndarray/Tensor
        第一组特征，形状为(N1, 64)。

    group2_features : list/ndarray/Tensor
        第二组特征，形状为(N2, 64)。

    standardize : bool
        是否在t-SNE之前对每个特征维度进行标准化。

    perplexity : float
        t-SNE困惑度，必须小于总样本数。

    random_state : int
        随机种子，保证结果可重复。

    save_path : str or None
        图片保存路径。

    Returns
    -------
    embedding : ndarray
        所有样本的二维t-SNE坐标，形状为(N1+N2, 2)。

    labels : ndarray
        每个样本对应的组别标签。
    """

    # 兼容PyTorch Tensor
    if hasattr(group1_features, "detach"):
        group1_features = (
            group1_features.detach().cpu().numpy()
        )

    if hasattr(group2_features, "detach"):
        group2_features = (
            group2_features.detach().cpu().numpy()
        )

    group1_features = np.asarray(
        group1_features,
        dtype=np.float64,
    )

    group2_features = np.asarray(
        group2_features,
        dtype=np.float64,
    )

    # 如果list中包含形状为(1, 64)的数组
    group1_features = np.squeeze(group1_features)
    group2_features = np.squeeze(group2_features)

    if group1_features.ndim == 1:
        group1_features = group1_features.reshape(1, -1)

    if group2_features.ndim == 1:
        group2_features = group2_features.reshape(1, -1)

    if group1_features.ndim != 2:
        raise ValueError(
            f"Group1特征必须是二维数组，"
            f"当前形状为{group1_features.shape}"
        )

    if group2_features.ndim != 2:
        raise ValueError(
            f"Group2特征必须是二维数组，"
            f"当前形状为{group2_features.shape}"
        )

    if (
            group1_features.shape[1]
            != group2_features.shape[1]
    ):
        raise ValueError(
            "两组特征的维度不同："
            f"{group1_features.shape[1]} vs "
            f"{group2_features.shape[1]}"
        )

    if group1_features.shape[1] != 64:
        print(
            f"提示：当前特征维度为"
            f"{group1_features.shape[1]}，不是64。"
        )

    # 合并两组特征
    features = np.concatenate(
        [group1_features, group2_features],
        axis=0,
    )

    labels = np.concatenate(
        [
            np.zeros(
                len(group1_features),
                dtype=int,
            ),
            np.ones(
                len(group2_features),
                dtype=int,
            ),
        ]
    )

    # 数据质量检查
    if not np.all(np.isfinite(features)):
        raise ValueError(
            "特征中存在NaN或Inf，请先清理数据。"
        )

    if len(features) < 4:
        raise ValueError(
            "t-SNE至少需要若干样本，"
            "当前总样本数过少。"
        )

    # 特征标准化
    if standardize:
        features_for_tsne = StandardScaler().fit_transform(
            features
        )
    else:
        features_for_tsne = features

    # perplexity必须小于总样本数
    effective_perplexity = min(
        float(perplexity),
        max(2.0, len(features) / 3.0),
    )

    if effective_perplexity >= len(features):
        effective_perplexity = len(features) - 1

    print("Group1 shape:", group1_features.shape)
    print("Group2 shape:", group2_features.shape)
    print("Total shape:", features.shape)
    print(
        "Effective perplexity:",
        effective_perplexity,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1000,  # n_iter
        random_state=random_state,
    )
    embedding = tsne.fit_transform(
        features_for_tsne
    )

    # 拆分二维坐标
    group1_embedding = embedding[
        labels == 0
        ]

    group2_embedding = embedding[
        labels == 1
        ]

    fig, ax = plt.subplots(
        figsize=(7 * 0.75, 6 * 0.75)
    )

    ax.scatter(
        group1_embedding[:, 0],
        group1_embedding[:, 1],
        label=group1_name,
        marker="o",
        s=30,
        alpha=0.75,
    )

    ax.scatter(
        group2_embedding[:, 0],
        group2_embedding[:, 1],
        label=group2_name,
        marker="^",
        s=35,
        alpha=0.75,
    )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("t-SNE visualization of hidden features")
    ax.legend(
        loc="best",
        frameon=True,
    )

    # t-SNE坐标轴本身没有直接物理含义
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return embedding, labels


def parse_str_or_int(value):
    try:
        # 尝试将输入转换为整数
        return int(value)
    except ValueError:
        # 如果转换失败，则保留为字符串
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-table',
                        default=r'LOSO_results12/BrainANet_SwinVIT.GAT.MDD.0_1.43/fold_21_test_S6/predictions_S6.csv')
    parser.add_argument('-weights',
                        default=r'LOSO_results12/BrainANet_SwinVIT.GAT.MDD.0_1.43/fold_21_test_S6/best_model.pth')
    parser.add_argument('-g', default=['0', '1'], nargs='+', help='Group', type=parse_str_or_int)  # ['CN', 'AD']
    parser.add_argument('-net', default='BrainANet_SwinVIT', type=str, help='BrainANet_SwinVIT,iRSSN')
    parser.add_argument('-task', default='MDD', type=str, help='Task,Dementia,ASD,MDD')
    args = parser.parse_args()

    test_id = pd.read_csv(args.table)['Subject ID'].tolist()
    frame_base = pd.read_csv(r'DataTable_classification.csv')

    # 风险ID
    path_danger_ids = r'cnn_correct_swin_wrong_results'
    danger_ids_list = []
    for file_name in os.listdir(path_danger_ids):
        pa_ = os.path.join(path_danger_ids, file_name, 'selected_subject_ids.csv')
        pa_frame = pd.read_csv(pa_)
        danger_ids_list.append(pa_frame)

    danger_ids_list = pd.concat(danger_ids_list)['Subject ID'].tolist()
    test_id = [i for i in test_id if i not in danger_ids_list]

    dict_ = {
        j: i for i, j in enumerate(args.g)
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = GAT(input_dim=load_dim(v=args.net, task=args.task), hidden=128, n_node=90)
    Net.load_state_dict(torch.load(args.weights, weights_only=False)['model_state_dict'])
    Net.eval()
    Net.to(device=device)

    list0, list1 = [], []
    with torch.no_grad():
        for id_ in test_id:
            subject_index = frame_base[frame_base['Subject ID'] == str(id_)].index[0]
            subject_information = frame_base.loc[subject_index, :]
            path_data = subject_information['Path_{}_node'.format(args.net)]
            data = pd.read_csv(path_data).values[:90, ...]
            node_features, edge_index, edge_weight, edge_attr = cosine_network_by_sparsity(data, alpha=0.85)
            label = dict_[subject_information['Group']]
            x_graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=torch.tensor(
                    [int(label)],
                    dtype=torch.long,
                ),

                site=str(subject_information["Stage"]),
                subject_id=str(subject_information["Subject ID"]),

            )
            x_graph.to(device=device)
            _, features = Net(x_graph, return_features=True)
            if label == 0:
                list0.append(features.cpu().detach().numpy())
            else:
                list1.append(features.cpu().detach().numpy())

    save_path = os.path.join('tsne_results1', r'tSNE.{}.{}_{}'.format(args.net, args.g[0], args.g[1]))
    os.makedirs(save_path, exist_ok=True)
    embedding, labels = plot_tsne_two_groups(
        group1_features=list0,
        group2_features=list1,
        group1_name=args.g[0],
        group2_name=args.g[1],
        standardize=True,
        perplexity=10,
        random_state=42,
        save_path=os.path.join(save_path, "tsne.eps"),
    )
