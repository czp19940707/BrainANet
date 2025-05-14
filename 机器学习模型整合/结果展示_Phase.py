import argparse
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

import random

random.seed(2023)


def extract_unique_parts(strings):
    # 找到所有字符串的公共前缀
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]  # 缩短前缀直到匹配

    # 找到所有字符串的公共后缀
    suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(suffix):
            suffix = suffix[1:]  # 缩短后缀直到匹配

    # 提取每个字符串中间的不同部分
    unique_parts = [s[len(prefix):-len(suffix)] if suffix else s[len(prefix):] for s in strings]
    return unique_parts


def check_results(model_list, fold_name):
    model_exist = []
    list_exist = os.listdir(fold_name)
    for model_name in model_list:
        if model_name in list_exist:
            model_exist.append(model_name)
        else:
            warnings.warn(f'{model_name} 不在results中, 已跳过！')
    return model_exist, extract_unique_parts(model_exist)


def generate_color(nums=100):
    list_color = []
    for i in range(nums):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # 排除红色或接近红色的颜色
        if not (r > 200 and g < 50 and b < 50):
            # return f'#{r:02x}{g:02x}{b:02x}'
            list_color.append(f'#{r:02x}{g:02x}{b:02x}')
    return list_color


def generate_colors_SCI():
    return ['#00468B', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A',
            '#ADB6B6', '#1B1919', '#374E55', '#DF8F44', '#00A1D5', '#B24745', '#79AF97',
            '#6A6599', '#80796B', '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
            '#849184', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']


def compute_metrics(y_true, y_pred, alpha=0.00000001):
    matrix = confusion_matrix(y_true, y_pred)
    TP = matrix[0][0]
    FN = matrix[0][1]
    FP = matrix[1][0]
    TN = matrix[1][1]

    accuracy = (TP + TN + alpha) / (TP + TN + FP + FN + alpha)
    sensitivity = (TP + alpha) / (TP + FN + alpha)
    specificity = (TN + alpha) / (TN + FP + alpha)
    precision = (TP + alpha) / (TP + FP + alpha)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    return accuracy, sensitivity, specificity, precision, f1


def show(args, fold_name, stage, Phase):
    save_path = os.path.join(fold_name, 'derivatives', args.s, stage)
    if stage == 'test':
        save_path = os.path.join(save_path, Phase)

    if os.path.exists(save_path):
        import sys
        print(
            f'{fold_name}/derivatives/{args.s}/{stage}文件夹已存在，请删除根目录下的{fold_name}/derivatives/{args.s}/{stage}文件夹，或者重新命名-s属性，确保结果不会被覆盖!')
        sys.exit()
    else:
        os.makedirs(save_path)

    method_list, legend_list = check_results([i.strip() for i in open(r'show_results.txt', 'r')], fold_name)

    plt.figure(figsize=(8, 6), dpi=300)  # 设置DPI为300
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color='red')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    color_selected = generate_colors_SCI()

    frame_save = pd.DataFrame()
    for nums, (method, legend_) in enumerate(zip(method_list, legend_list)):
        frame_results = pd.read_csv(os.path.join(fold_name, method, f'result_{stage}.csv'))
        frame_results = frame_results[frame_results['Phase'] == Phase]
        pred = frame_results['Pred'].values
        prob = frame_results[f'Prob_{1}'].values
        cls = frame_results['Group'].values
        fpr, tpr, threshold = roc_curve(cls, prob, pos_label=1)
        acc, sen, spe, pre, f1 = compute_metrics(cls, pred)
        roc_auc = auc(fpr, tpr)
        frame_save.loc[nums, 'method'] = legend_
        frame_save.loc[nums, 'Accuracy'] = f"{(acc * 100):.3g}"
        frame_save.loc[nums, 'Sensitivity'] = f"{(sen * 100):.3g}"
        frame_save.loc[nums, 'Specificity'] = f"{(spe * 100):.3g}"
        frame_save.loc[nums, 'Precision'] = f"{(pre * 100):.3g}"
        frame_save.loc[nums, 'F1'] = f"{(f1 * 100):.3g}"
        frame_save.loc[nums, 'ROC'] = f"{(roc_auc * 100):.3g}"

        # plt.plot(fpr, tpr, label=method.split('.')[0].split('_ridge_')[-1] + '_' + method.split('.')[1].split('_')[0] + '_' + method.split('.')[-1].split('_')[0] + ': {}'.format(str(roc_auc)[:6]), lw=2, alpha=.8, color=color_selected[nums])
        plt.plot(fpr, tpr,
                 label=legend_ + ': {}'.format(str(roc_auc)[:6]), lw=2, alpha=.8,
                 color=color_selected[nums])

    plt.grid(True)
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right', ncol=1)
    plt.savefig(os.path.join(save_path, f'roc.jpg'), dpi=300)
    plt.close()

    """
    显示柱状图
    """
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
    metrics = frame_save.columns.tolist()[1:]
    indexes = frame_save.index.tolist()

    x_sub = [(i - len(indexes) // 2) * bar_width for i in range(len(indexes))]

    for i, index in enumerate(indexes):
        for j, metric in enumerate(metrics):
            position = j + x_sub[i]
            value = float(frame_save.loc[index, metric])
            plt.bar(position, value, bar_width, label=frame_save.loc[index, 'method'] if j == 0 else '',
                    alpha=0.8,
                    color=color_selected[i])

    ax.set_xticks([i for i in range(len(metrics))], metrics)
    ax.set_ylim(frame_save.loc[:, metrics].astype(np.float32).values.min() - 5,
                frame_save.loc[:, metrics].astype(np.float32).values.max() + 10)
    ax.grid(True)
    ax.legend(loc='best', ncol=len(indexes))
    plt.savefig(os.path.join(save_path, f'bar.jpg'), dpi=300)
    plt.close()

    """
    绘制雷达图
    """
    metrics = frame_save.columns.tolist()[1:]
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for nums, index_ in enumerate(frame_save.index):
        values = (frame_save.loc[index_, :].values[1:].astype(np.float32) / 100).tolist()
        category = frame_save.loc[index_, 'method']
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, label=category, linewidth=2, color=color_selected[nums], marker='o', markersize=6)
        ax.fill(angles, values, color=color_selected[nums], alpha=0.1)  # 添加透明填充

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.grid(color='gray', linestyle='--', linewidth=0.8)
    ax.spines['polar'].set_visible(False)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10, fontweight='bold', color='black')

    max_value, min_value = frame_save.values[:, 1:].astype(np.float32).max(), frame_save.values[:, 1:].astype(
        np.float32).min()
    y_ = np.linspace(np.floor(min_value - 3) / 100, np.ceil(max_value + 3) / 100, 4, endpoint=True).tolist()
    y_str = [str(i)[:3] for i in y_]
    ax.set_yticks(y_)
    ax.set_yticklabels(y_str, color="gray", size=10)
    ax.set_ylim(np.min(y_), np.max(y_))

    ax.set_title(args.s, size=12, pad=20, weight='bold')

    # 调整图例位置
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(method_list), fontsize=10, frameon=False)

    plt.subplots_adjust(top=0.85, bottom=0.25)  # 调整边距以适应图例

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'radar.jpg'), dpi=300)
    plt.close()

    """
    Table结果保存
    """
    frame_save.to_csv(os.path.join(save_path, f'metrics.csv'), index=False)


def show_metrix_polt(train_frame: pd.DataFrame, test_frame: pd.DataFrame, fold_name):
    """
    灰质ACC_SEN_SPE_AUC子图折线图
    """
    save_path = os.path.join(fold_name, 'derivatives', args.s, 'combine')
    if os.path.exists(save_path):
        import sys
        print(
            f'{fold_name}/derivatives/{args.s}/combine文件夹已存在，请删除根目录下的{fold_name}/derivatives/{args.s}/combine文件夹，或者重新命名-s属性，确保结果不会被覆盖!')
        sys.exit()
    else:
        os.makedirs(save_path)
    metrics = train_frame.columns.tolist()[1:]
    fig, axs = plt.subplots(figsize=(2.2 * len(metrics), 2.2), dpi=300)

    gs = gridspec.GridSpec(1, len(metrics), width_ratios=[1 for _ in range(len(metrics))])
    x = [i for i in range(len(train_frame['method'].values.tolist()))]
    x_ticks = ['{}*{}*{}'.format(i, i, i) for i in train_frame['method'].values.tolist()]
    for i, metric in enumerate(metrics):
        train_frame_sub, test_frame_sub = train_frame[metric].astype(float).values.tolist(), test_frame[metric].astype(
            float).values.tolist()
        ax = plt.subplot(gs[i])
        ax.plot(x, train_frame_sub, marker='o', color='blue', linestyle='dotted', label='Train' if i == 0 else '',
                linewidth=3)
        ax.plot(x, test_frame_sub, marker='s', color='green', linestyle='-', label='Test' if i == 0 else '',
                linewidth=3)
        ax.set_ylabel(metric)
        # ax.set_xlabel('Patch-size')
        ax.set_xticks(x, x_ticks, fontsize=7)
        # ax.set_xticklabels(x, fontsize=12)  # 设置字体大小为12
        ax.tick_params(axis='x', rotation=45)
        # 省略右侧和上侧坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.legend(loc='upper center', ncol=2, fontsize='large', frameon=False)
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 调整顶部空间
    plt.savefig(os.path.join(save_path, 'plot.jpg'), dp1=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default='results_MDD_1', type=str, help='Task name')
    parser.add_argument('-s', default='Sparsity compare CN vs. MDD', type=str, help='想要对比什么，起个名字')
    parser.add_argument('-stage', default='test', type=str, help='train or test')
    parser.add_argument('-Phase', default='S19', type=str, help='dataset name')
    args = parser.parse_args()

    show(args, stage=args.stage, fold_name=args.t, Phase=args.Phase)
    # show(args, stage='test', fold_name=args.t)