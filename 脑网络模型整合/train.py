import argparse
from utils import parse_str_or_int, load_net, group_to_str, preprocess, model_forward_train, model_forward_eval
import time
from torch import nn, optim
import torch
import os
from dataProc import dataProc
import pandas as pd
import torch.nn.functional as F


def train():
    global fold
    net.train()
    time_start = time.time()
    train_loss = 0.0
    for batch_idx, (data, label, _) in enumerate(train_loader):
        # data = data.to(device)
        data = preprocess(data, label, threshold=args.threshold, sparsity_method=args.sparsity)
        # label = label.to(device)
        optimizer.zero_grad()
        # out = net.forward(data.to(device), isExplain=args.isExplain)
        # loss = loss_function(out, label)
        loss = model_forward_train(args, net, data, device)
        train_loss += loss
        loss.backward()
        optimizer.step()

    net.loss_dict['train_loss'].append(train_loss.item())
    time_end = time.time()
    print('*' * 30 + 'Training Finish' + '*' * 30)
    print(
        f'Epoch: {epoch + 1}\tTrain loss: {train_loss.item()}\tTrain Mean Loss: {train_loss.item() / len(train_set)}\tTime: {time_end - time_start}\tLR: {optimizer.param_groups[0]["lr"]}')
    print('*' * 30 + '***************' + '*' * 30)


def val():
    global best_acc
    global fold
    global result_dict, params_dict
    net.eval()
    time_start = time.time()
    eval_loss, correct = 0.0, 0.0
    index_list, label_list, preds_list, probs_list, loss_list = [], [], [], [], []
    with torch.no_grad():
        for data, label, index in val_loader:
            data = preprocess(data, label, threshold=args.threshold, sparsity_method=args.sparsity)
            label = label.to(device)
            # data = data.to(device)
            out, loss, loss_nll = model_forward_eval(args, net, data.to(device), device)

            eval_loss += loss

            """
            保存结果
            """
            _, preds = out.max(1)
            index_list.append(index)
            label_list.append(label)
            probs_list.append(F.softmax(out, dim=1))
            loss_list.append(loss_nll)
            preds_list.append(preds)

            """
            计算准确度作为模型保存的标准
            """
            correct += out.max(1)[-1].eq(label).sum()

    net.loss_dict['eval_loss'].append(eval_loss.item())
    time_end = time.time()
    acc = correct.float() / len(val_set)

    if acc >= best_acc:
        result_dict = {
            'index': torch.cat(index_list, dim=0),
            'label': torch.cat(label_list, dim=0),
            'probs': torch.cat(probs_list, dim=0),
            'loss': torch.cat(loss_list, dim=0),
            'preds': torch.cat(preds_list, dim=0),

        }
        best_acc = acc
        if args.save_params:
            params_dict = {k: v.clone().detach() for k, v in net.state_dict().items()}

    print('*' * 30 + 'Validation Finish' + '*' * 30)
    print(
        f'Epoch: {epoch + 1}\tEval loss: {eval_loss.item()}\tEval Mean Loss: {eval_loss.item() / len(val_set)}\tTime: {time_end - time_start}\tAccuracy: {acc}')
    print('*' * 30 + '***************' + '*' * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', default='SGCN_GAT', type=str, help='模型版本')
    parser.add_argument('-net', default='SwinVIT_D32_P24', type=str, help='相似性网络格式，iRSSN或者Sem,SwinVIT_D32_P24+rest_fMRI_aal116')
    parser.add_argument('-g', default=[0, 1], nargs='+', type=parse_str_or_int, help='Group')  # ['CN', 'AD']
    parser.add_argument('-e', default=100, type=int, help='Epochs')
    parser.add_argument('-b', default=16, type=int, help='Batch size')
    parser.add_argument('-frame', default=r'DataTable_MDD2.csv', type=str)
    parser.add_argument('-save_params', default=False, type=bool)
    parser.add_argument('-threshold', default=0.4, type=float, help='Threshold')
    # parser.add_argument('-isExplain', default=True, type=bool)
    parser.add_argument('-sparsity', default='percent', type=str)
    parser.add_argument('-s', default='result_fc_BrainANet', type=str, help='保存路径')
    args = parser.parse_args()

    net = load_net(args)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    save_path = os.path.join(args.s,
                             f'{args.v}.{args.sparsity}_{args.threshold}.{args.net}.{group_to_str(args.g)}')
    os.makedirs(save_path, exist_ok=True)
    if args.save_params:
        params_dict = {}

    train_set = dataProc(stage='train', group=args.g, frame_path=args.frame, net=args.net)
    sample_weight, imbalanced_ratio = train_set.get_sample_weight()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.b, sampler=sampler, num_workers=8,
                                               drop_last=True)

    val_set = dataProc(stage='test', group=args.g, frame_path=args.frame, net=args.net)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.b, shuffle=False, num_workers=8, drop_last=False)

    best_acc = 0.0
    result_dict = {}
    for epoch in range(args.e):
        train()
        val()

    """
    结果保存
    """
    frame = train_set.frame
    frame_save = pd.DataFrame()
    count_ = 0
    for index, label, prob, loss, pred in zip(result_dict['index'],
                                              result_dict['label'],
                                              result_dict['probs'],
                                              result_dict['loss'],
                                              result_dict['preds']):
        frame_save.loc[count_, 'Subject ID'] = frame.loc[index.item(), 'Subject ID']
        frame_save.loc[count_, 'Phase'] = frame.loc[index.item(), 'Phase']
        frame_save.loc[count_, 'Group'] = label.item()
        frame_save.loc[count_, 'Pred'] = pred.item()
        frame_save.loc[count_, 'Loss'] = loss.item()
        for n, pr in enumerate(prob):
            frame_save.loc[count_, f'Prob_{n}'] = pr.item()

        count_ += 1

    frame_save.to_csv(os.path.join(save_path, r'result_test.csv'), index=False)

    """
    保存loss
    """
    import pickle

    with open(os.path.join(save_path, r'loss.pkl'), 'wb') as f:
        pickle.dump(net.loss_dict, f)

    """
    保存模型参数
    """
    if args.save_params:
        torch.save(params_dict, os.path.join(save_path, r'params.pth'))
