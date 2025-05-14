import pandas as pd
import argparse
from model import My_SSLHead
import torch
import nibabel as nib
from torch import nn
import numpy as np
import os


def crop_patch(image, center, patch_size):
    """
    根据中心点 (x, y, z) 和 patch 大小裁剪图像。

    参数:
    - image: 3D 图像数据，NumPy 数组
    - center: 中心坐标 (x, y, z)
    - patch_size: patch 的大小 (例如 16 表示 16x16x16)

    返回:
    - patch: 从图像中裁剪的 patch
    """

    # 获取图像尺寸
    img_shape = image.shape

    # 计算半个 patch 大小
    half_size = patch_size // 2

    # 计算每个维度的起始和结束坐标
    x, y, z = center
    start_x = max(0, x - half_size)
    end_x = min(img_shape[0], x + half_size)

    start_y = max(0, y - half_size)
    end_y = min(img_shape[1], y + half_size)

    start_z = max(0, z - half_size)
    end_z = min(img_shape[2], z + half_size)

    # 如果 patch_size 是偶数，确保裁剪后为正确尺寸
    if (end_x - start_x) != patch_size:
        end_x -= 1
    if (end_y - start_y) != patch_size:
        end_y -= 1
    if (end_z - start_z) != patch_size:
        end_z -= 1

    # 裁剪出 patch
    patch = image[start_x:end_x, start_y:end_y, start_z:end_z]

    return patch[None, None, ...]


def padding(patch):
    _, _, w, h, d = patch.shape
    if not all(v >= args.p for v in [w, h, d]):
        mask = np.zeros([1, 1, args.p, args.p, args.p])
        mask[:, :, :w, :h, :d] = patch
        patch = mask
    return patch


# /media/shucheng/数程SSD_2T/PPMI/cat12/mri/mwp1162598.nii
# /media/shucheng/数程SSD_2T/MDD/derivatives/cat12/mri/S3-2-0034.nii

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', default='CNN', type=str, help='模型名称,SwinVIT_D64_P24')
    parser.add_argument('-i', default='/media/shucheng/数程SSD_2T/ABIDE/derivatives/cat12/mri/mwp128680_anat.nii')
    parser.add_argument('-s', default=r'/media/shucheng/数程SSD_2T/ABIDE/derivatives')
    #parser.add_argument('-subject_id', default=r'mwp1subj05_1A')
    parser.add_argument('-p', default=24, type=int)
    parser.add_argument('-dim', default=192, type=int, help='模型深度')
    parser.add_argument('-dim_hidden', default=32, type=int, help='提取深度特征维度')
    args = parser.parse_args()

    args.spatial_dims = 3
    args.in_channels = 1
    args.feature_size = 24
    args.dropout_path_rate = 0.2
    args.use_checkpoint = False
    args.local_rank = 0  # 加载到的cuda设备，0或者cup
    args.subject_id = os.path.split(args.i)[-1].replace('.nii', '')

    net = My_SSLHead(args=args, dim=args.dim, net_name=args.v, dim_hidden=args.dim_hidden)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(f'weights9/{args.v}_D{args.dim_hidden}_P{args.p}/10.pth', map_location=device))
    net.to(device)

    patch_size = args.p

    rois_frame = pd.read_csv(r'Reslice_AAL116_center_locate_15mm.csv')

    data = nib.load(args.i).get_fdata()
    net.eval()
    feat_list = []

    with torch.no_grad():
        for index in rois_frame.index.tolist()[1:]:
            roi_information = rois_frame.loc[index, :]
            x, y, z = roi_information['X'], roi_information['Y'], roi_information['Z']
            patch = crop_patch(data, center=(x, y, z), patch_size=patch_size)
            patch = padding(patch)
            patch = torch.from_numpy(patch).float().to(device)
            patch_feat = net.forward(patch)[1].cpu().numpy()
            # patch_feat = nn.AdaptiveAvgPool3d(1)(patch_feat)
            # patch_feat = nn.Flatten()(patch_feat).cpu().numpy()
            feat_list.append(patch_feat)

    frame_save = pd.DataFrame(np.concatenate(feat_list, axis=0))
    frame_save = (frame_save - frame_save.mean(axis=0)) / frame_save.std(axis=0)
    iDRSN = frame_save.T.corr(method='pearson')
    save_path = os.path.join(args.s, f'{args.v}_D{args.dim_hidden}_P{args.p}', 'data')
    os.makedirs(save_path, exist_ok=True)
    frame_save.to_csv(os.path.join(save_path, f'Sem_{args.subject_id}.csv'), index=False)
    iDRSN.to_csv(os.path.join(save_path, f'iDRSN_{args.subject_id}.csv'), index=False)
