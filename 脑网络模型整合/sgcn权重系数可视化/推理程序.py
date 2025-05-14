import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from dataProc import dataProc
from utils import load_net, parse_str_or_int, preprocess
from scipy.stats import ttest_ind

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=str, default='SGCN_GAT', help='model name')
    parser.add_argument('-net', type=str, default='SwinVIT_D32_P24', help='SwinVTI, CNN, iRSSN, R2SN')
    # parser.add_argument('-param_path',
    #                     default='/home/shucheng/python_files/深度学习模型/脑网络模型整合/results_MDD/SGCN_GAT.percent_0.2.SwinVIT.0_1/params.pth',
    #                     type=str, help='参数路径')
    parser.add_argument('-g', default=['sMCI', 'pMCI'], nargs='+', type=parse_str_or_int, help='Group')  # ['CN', 'AD']
    parser.add_argument('-fold', default='result_save_param', type=str)
    parser.add_argument('-sparsity_method', default='percent', type=str)
    parser.add_argument('-threshold', default=0.8, type=str)

    parser.add_argument('-frame', default=r'../DataTable_AD.csv', type=str)
    parser.add_argument('-topk', default=100, type=int)
    parser.add_argument('-s', default='results', type=str)
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    net = load_net(args)
    args.param_path = f'../{args.fold}/{args.v}.{args.sparsity_method}_{args.threshold}.{args.net}.{args.g[0]}_{args.g[1]}/params.pth'
    try:
        net.load_state_dict(torch.load(args.param_path))
    except:
        raise ValueError('参数和模型不匹配！')
    net.to(device)
    net.eval()

    data_set = dataProc(stage='train', group=args.g, frame_path=args.frame, net=args.net)

    frame = pd.DataFrame()

    group1, group2 = [], []
    with torch.no_grad():
        for data, label, index_ in data_set:
            # label = label.to(device)
            data = torch.tensor(data)
            data_graph = preprocess(data[None, ...], np.array(label)[None, ...], args.threshold, args.sparsity_method)
            data_graph = data_graph.to(device)
            x, edge_index, batch, edge_weight = data_graph.x, data_graph.edge_index, data_graph.batch, data_graph.edge_attr
            # _, _, _, edge_prob = net.cal_probability(x, edge_index, edge_weight)
            N, D = x.shape
            x = x.reshape(N // net.rois, net.rois, D)
            x_prob = net.prob
            x_feat_prob = x * x_prob
            x_feat_prob = x_feat_prob.reshape(N, D)
            conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
            edge_prob = torch.sigmoid(conat_prob.matmul(net.prob_bias)).view(-1)

            edge_prob_matrix = torch.zeros((N, N)).to(device)
            row_index = edge_index[0, :]
            col_index = edge_index[1, :]
            edge_prob_matrix[row_index, col_index] = edge_prob
            if label == 0:
                group1.append(edge_prob_matrix.cpu().numpy().flatten()[None, ...])
            elif label == 1:
                group2.append(edge_prob_matrix.cpu().numpy().flatten()[None, ...])

    group1 = np.concatenate(group1, axis=0)
    group2 = np.concatenate(group2, axis=0)
    stat, p_values = ttest_ind(group1, group2, axis=0)

    topk = np.argsort(p_values)[:args.topk]

    dict_ = {}
    count_ = 0
    for i in range(90):
        for j in range(90):
            dict_[count_] = str(i) + '_' + str(j)
            count_ += 1

    count_ = 0
    for idx in topk:
        frame.loc[count_, 'Feature Name'] = dict_[idx]
        frame.loc[count_, 'coef'] = np.abs(stat[idx])
        count_ += 1

    save_path = os.path.join(args.s, f'{args.v}.{args.sparsity_method}_{args.threshold}.{args.net}.{args.g[0]}_{args.g[1]}')
    os.makedirs(save_path, exist_ok=True)
    frame.to_csv(os.path.join(save_path, f'coef_top{args.topk}.csv'), index=False)