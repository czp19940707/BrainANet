import argparse

import numpy as np

from dataProc import dataProc, rot_rand, aug_rand
import torch
import torch_geometric
from model import My_SSLHead, Loss
import time
from torch import nn, optim
import os
from base_model import count_parameters

def check_zero_batch(data):
    return torch.all(data == 0)


def train():
    global f_
    net.train()
    time_start = time.time()
    train_loss = 0.0
    for batch_idx, (data, data_index) in enumerate(train_loader):
        data = data.to(device)
        x1, rot1 = rot_rand(args, data)
        x2, rot2 = rot_rand(args, data)
        x1_augment = aug_rand(args, x1)
        x2_augment = aug_rand(args, x2)

        if check_zero_batch(x1_augment) or check_zero_batch(x2_augment):
            error_message = '整个batch全是0，请核查: ' + ','.join(
                [str(i) for i in data_index.cpu().detach().numpy()])
            f_.write(error_message + '\n')
            continue  # 跳过全为 0 的批次

        rot1_p, contrastive1_p, recon1_p = net(x1_augment)
        rot2_p, contrastive2_p, recon2_p = net(x2_augment)

        imgs_recon = torch.cat([recon1_p, recon2_p], dim=0)
        imgs = torch.cat([x1, x2], dim=0)

        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        # imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        # imgs = torch.cat([x1, x2], dim=0)
        loss, _ = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
        if not torch.isnan(loss).any():
            optimizer.zero_grad()
            train_loss += loss
            loss.backward()
            optimizer.step()
        else:
            error_message = '出现错误的索引可能是其中之一，请核查: ' + ','.join(
                [str(i) for i in data_index.cpu().detach().numpy()])
            f_.write(error_message + '\n')

    net.loss_dict['train_loss'].append(train_loss.item())
    time_end = time.time()
    print('*' * 30 + 'Training Finish' + '*' * 30)
    print(
        f'Epoch: {epoch + 1}\tTrain loss: {train_loss.item()}\tTrain Mean Loss: {train_loss.item() / len(train_set)}\tTime: {time_end - time_start}')
    print('*' * 30 + '***************' + '*' * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', default='CNN', type=str, help='模型名称')
    # parser.add_argument('-voxel_size', default='15mm', type=str)
    parser.add_argument('-e', default=151, type=int, help='Epochs')
    parser.add_argument('-b', default=8, type=int, help='Batch size')
    parser.add_argument('-patch_size', default=16, type=int, help='Gen size')
    parser.add_argument('-path', default=r'DataTable.csv', type=str, help='数据管理表格路径（e.g. Amsterdam.csv）')
    parser.add_argument('-mask_path', default=r'/media/shucheng/数程SSD_2T/Reslice_AAL116_1mm_15mm.nii', type=str)
    parser.add_argument('-dim', default=192, type=int, help='模型深度')
    parser.add_argument('-dim_hidden', default=32, type=int, help='提取深度特征维度')

    args = parser.parse_args()

    train_set = dataProc(patch_size=args.patch_size, table_path=args.path, mask_path=args.mask_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.b, shuffle=True, num_workers=8,
                                               drop_last=True, collate_fn=train_set.my_collate_fn)

    """
    记录错误信息
    """
    save_path = f'weights9/{args.v}_D{args.dim_hidden}_P{args.patch_size}'
    os.makedirs(save_path, exist_ok=True)
    f_ = open(os.path.join(save_path, 'error_message.txt'), 'w')

    args.spatial_dims = 3
    args.in_channels = 1
    args.feature_size = 24
    args.dropout_path_rate = 0.2
    args.use_checkpoint = False
    args.local_rank = 0  # 加载到的cuda设备，0或者cup

    net = My_SSLHead(args, dim=args.dim, net_name=args.v, dim_hidden=args.dim_hidden)
    # print(count_parameters(net))    # (1312039)
    # net.load_state_dict(torch.load(r'weights2/CNN_16/10.pth'), strict=False)

    device = net.device
    net.to(device)

    loss_function = Loss(args)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    best_acc = 0.0
    for epoch in range(args.e):
        train()
        if epoch % 50 == 0:
            torch.save(net.state_dict(), os.path.join(save_path, f'{epoch}.pth'))
    loss_store_list = net.loss_dict['train_loss']
    os.makedirs('Loss', exist_ok=True)
    np.save(os.path.join('Loss', args.v + f'_{args.patch_size}.npy'), np.array(loss_store_list))
