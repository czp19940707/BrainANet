import os

import pandas as pd
from torch.utils.data import Dataset
import shutil
import glob
import nibabel as nib
import numpy as np
import random
import torch
import numpy as np
from numpy.random import randint
from monai.transforms import Compose, RandRotate90, RandFlip, RandGaussianNoise, ToTensor


def generate_path_list():
    path_HCP = r'/media/shucheng/蓝硕32T/HCP/derivatives/cat12/mri'
    save_path = r'/media/shucheng/数程SSD_2T/HCP/derivatives/cat12/mri'
    os.makedirs(save_path, exist_ok=True)
    dataset_frame = pd.DataFrame()
    # for index_, filename in enumerate(os.listdir(path_HCP)):
    #     if filename.startswith('mwp1'):
    #         dataset_frame.loc[index_, 'Dataset'] = 'HCP'
    #         dataset_frame.loc[index_, 'Subject ID'] = 'sub-' + filename.split('_')[0].split('-')[-1]
    #         dataset_frame.loc[index_, 'Path'] = os.path.join(save_path, filename)
    #         shutil.copyfile(os.path.join(path_HCP, filename), os.path.join(save_path, filename))

    path_Amsterdam1 = r'/media/shucheng/蓝硕32T/OpenNeuro/ds002790/derivatives/cat12/mri'
    path_Amsterdam2 = r'/media/shucheng/蓝硕32T/OpenNeuro/ds003097/derivatives/cat12/mri'
    path_Amsterdam3 = r'/media/shucheng/蓝硕32T/OpenNeuro/ds002785/derivatives/cat12/mri'

    nii_files_Amsterdam1 = glob.glob(os.path.join(path_Amsterdam1, '**', 'mwp1*'), recursive=True)
    nii_files_Amsterdam2 = glob.glob(os.path.join(path_Amsterdam2, '**', 'mwp1*'), recursive=True)
    nii_files_Amsterdam3 = glob.glob(os.path.join(path_Amsterdam3, '**', 'mwp1*'), recursive=True)

    save_path_Amsterdam1 = r'/media/shucheng/数程SSD_2T/Amsterdam1/derivatives/cat12/mri'
    save_path_Amsterdam2 = r'/media/shucheng/数程SSD_2T/Amsterdam2/derivatives/cat12/mri'
    save_path_Amsterdam3 = r'/media/shucheng/数程SSD_2T/Amsterdam3/derivatives/cat12/mri'

    os.makedirs(save_path_Amsterdam1, exist_ok=True)
    os.makedirs(save_path_Amsterdam2, exist_ok=True)
    os.makedirs(save_path_Amsterdam3, exist_ok=True)

    for index_, file_path in enumerate(nii_files_Amsterdam1):
        dataset_frame.loc[index_, 'Dataset'] = 'Amsterdam1'
        dataset_frame.loc[index_, 'Subject ID'] = os.path.split(file_path)[-1].split('_')[0]
        dataset_frame.loc[index_, 'Path'] = os.path.join(save_path_Amsterdam1, os.path.split(file_path)[-1])
        # shutil.copyfile(file_path, os.path.join(save_path_Amsterdam1, os.path.split(file_path)[-1]))

    for index_, file_path in enumerate(nii_files_Amsterdam2):
        dataset_frame.loc[index_, 'Dataset'] = 'Amsterdam2'
        dataset_frame.loc[index_, 'Subject ID'] = os.path.split(file_path)[-1].split('_')[0]
        dataset_frame.loc[index_, 'Path'] = os.path.join(save_path_Amsterdam2, os.path.split(file_path)[-1])
        # shutil.copyfile(file_path, os.path.join(save_path_Amsterdam2, os.path.split(file_path)[-1]))

    for index_, file_path in enumerate(nii_files_Amsterdam3):
        dataset_frame.loc[index_, 'Dataset'] = 'Amsterdam3'
        dataset_frame.loc[index_, 'Subject ID'] = os.path.split(file_path)[-1].split('_')[0]
        dataset_frame.loc[index_, 'Path'] = os.path.join(save_path_Amsterdam3, os.path.split(file_path)[-1])
        # shutil.copyfile(file_path, os.path.join(save_path_Amsterdam3, os.path.split(file_path)[-1]))

    dataset_frame.to_csv('Amsterdam.csv', index=False)


class dataProc(Dataset):
    def __init__(self, patch_size=16, table_path='DataTable.csv', mask_path=r'AAL116_1mm_15mm_mask.nii.gz'):
        self.mask_path = mask_path
        self.frame = pd.read_csv(table_path)
        self.patch_size = patch_size
        self.valid_locates = self.load_brainMask()
        self.index = self.frame.index
        # self.transform_pipeline = Compose([
        #     RandRotate90(prob=0.5),  # 随机旋转90度
        #     RandFlip(prob=0.5, spatial_axis=0),  # 随机水平翻转
        #     RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),  # 随机添加高斯噪声
        #     ToTensor()  # 转换为Tensor格式
        # ])
        # self.check_shape = 16



    def __len__(self):
        return len(self.index)

    def load_brainMask(self):
        mask = nib.load(self.mask_path).get_fdata()
        self.briainMask = np.zeros_like(mask, dtype=np.uint8)
        self.briainMask[mask != 0] = 1
        brain_x, brain_y, brain_z = self.briainMask.shape
        locate = np.where(self.briainMask == 1)
        valid_locates = []
        half_patch = self.patch_size // 2
        for coord in np.array(locate).T:
            x, y, z = coord
            if (x - half_patch >= 0 and x + half_patch < brain_x and
                    y - half_patch >= 0 and y + half_patch < brain_y and
                    z - half_patch >= 0 and z + half_patch < brain_z):
                valid_locates.append((x - half_patch, y - half_patch, z - half_patch))

        return valid_locates

    def z_score(self, image):
        mean = np.sum(self.briainMask * image) / np.sum(self.briainMask)
        variance = np.sum((image - mean) ** 2) / (np.sum(self.briainMask))
        return (image - mean) / variance

    # def label_transform(self, data):
    #     data = torch.tensor(data, dtype=torch.float32)[None, ...]
    #     data_transformed = self.transform_pipeline(data)
    #     return data, data_transformed

    def check_data(self, data):
        data_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size))
        while (len(np.where(data_patch != 0)[0]) / (self.patch_size * self.patch_size * self.patch_size)) < 0.2:
            center_locate = random.choice(self.valid_locates)
            data_patch = data[center_locate[0]: center_locate[0] + self.patch_size,
                         center_locate[1]: center_locate[1] + self.patch_size,
                         center_locate[2]: center_locate[2] + self.patch_size]
        return data_patch

    def __getitem__(self, index):
        data = nib.load(self.frame.loc[self.index[index], 'Path']).get_fdata()
        z_data = self.z_score(data)
        data_patch = self.check_data(z_data)
        # data_patch, label_patch = self.label_transform(data_patch)
        # return data_patch, label_patch, self.index[index]
        # if data_patch.shape[0] != 16:
        #     print(self.frame.loc[self.index[index], 'Path'])
        return data_patch[None, ...].astype(np.float32), self.index[index]

    def my_collate_fn(self, batch):
        data_, index_ = [], []
        for data in batch:
            n, w, h, d = data[0].shape
            if not all(v >= self.patch_size for v in [w, h, d]):
                mask = np.zeros([n, self.patch_size, self.patch_size, self.patch_size])
                mask[:, :w, :h, :d] = data[0]
                data[0] = mask
            data_.append(torch.tensor(data[0]))
            index_.append(torch.tensor(data[1]))
        return torch.stack(data_), torch.tensor(index_)


def rot_rand(args, x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug


def patch_rand_drop(args, x, x_rep=None, max_drop=0.25, max_block_sz=0.2, tolr=0.03):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix or torch.all(x == 0):

        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s) <= 1:
            continue
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=args.local_rank
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                    torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )

            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x


def custom_collate_fn(batch):
    """
    自定义批次组装逻辑
    Args:
    - batch (list of tuples): 每个样本是 (data, label) 的形式
    Returns:
    - data_tensor: 批次数据
    - label_tensor: 批次标签
    """
    data, labels = zip(*batch)
    data_tensor = torch.stack(data)  # 将数据堆叠为一个张量
    label_tensor = torch.tensor(labels)
    return data_tensor, label_tensor


# class my_collate_fn:
#     def __init__(self, args):
#         self.patch_size = args.patch_size
#
#     def __call__(self, batch):
#         for data, label in zip(*batch):
#             print()

if __name__ == '__main__':
    dp = dataProc()
    for i in dp:
        print()
    # generate_path_list()
