# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    def __init__(self, args, dim=768, dim_hidden=32):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(1, args.spatial_dims)  # 2 (原始)
        window_size = ensure_tuple_rep(3, args.spatial_dims)  # 7 (元素)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],  # depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],  # num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, dim_hidden)

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x4_reshape = nn.AdaptiveAvgPool1d(2)(x4_reshape.transpose(1, 2))
        x_rot = self.rotation_pre(x4_reshape[..., 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[..., 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_rot, x_contrastive, x_rec


SSLHead_selected = SSLHead

"""
conventional CNN


"""


class CNN_backbone(nn.Module):
    def __init__(self, dim=32, dim_hidden=32):
        super(CNN_backbone, self).__init__()
        # self.swinViT = nn.ModuleDict()
        # self.swinViT['layer1'] = nn.Sequential(
        #     nn.Conv3d(1, 8, 3, padding=1),
        #     nn.BatchNorm3d(8),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.MaxPool3d(2, 2),
        # )
        # self.swinViT['layer2'] = nn.Sequential(
        #     nn.Sequential(
        #         nn.Conv3d(8, 16, 3, padding=1),
        #         nn.BatchNorm3d(16),
        #         nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #         nn.MaxPool3d(2, 2),
        #     )
        # )
        # self.swinViT['layer3'] = nn.Sequential(
        #     nn.Sequential(
        #         nn.Conv3d(16, 32, 3, padding=1),
        #         nn.BatchNorm3d(32),
        #         nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #         nn.MaxPool3d(2, 2),
        #     )
        # )

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, dim // 4, 3, padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool3d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, 3, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool3d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, 3, padding=1),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool3d(2, 2),
        )

        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, dim_hidden)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return [None, x1, x2, x3, None]


class My_SSLHead:
    def __init__(self, args, dim=768, net_name='SwinVIT', dim_hidden=32):
        # 动态选择父类
        self.net = net_name
        if net_name.startswith('SwinVIT'):
            ParentClass = SSLHead
            NewClass = type("My_SSLHead", (ParentClass,), {"forward": self.forward})
            self.__class__ = NewClass
            ParentClass.__init__(self, args, dim=dim, dim_hidden=dim_hidden)
        elif net_name.startswith('CNN'):
            ParentClass = CNN_backbone
            NewClass = type("My_SSLHead", (ParentClass,), {"forward": self.forward, "swinViT": CNN_backbone.forward})
            self.__class__ = NewClass
            ParentClass.__init__(self, dim=dim, dim_hidden=dim_hidden)
        else:
            raise ValueError("Invalid condition! Choose 'SwinVIT' or 'CNN'.")

        # 动态创建子类

        if net_name.startswith('SwinVIT'):
            self.swinViT.layers4 = nn.ModuleList()
            for i in range(2):
                # self.swinViT.layers3.append(nn.Identity())
                self.swinViT.layers4.append(nn.Identity())

        self.deconv1 = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv3d(dim, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )

        self.conv = nn.Conv3d(dim // 8, args.in_channels, kernel_size=1, stride=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_dict = {
            'train_loss': [],
            'eval_loss': [],
        }

    def forward(self, x):

        x_1, x_2, x_3 = self.swinViT(x.contiguous())[1:4]

        _, c, h, w, d = x_3.shape
        x4_reshape = x_3.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x4_reshape = nn.AdaptiveAvgPool1d(2)(x4_reshape.transpose(1, 2))
        x_rot = self.rotation_pre(x4_reshape[..., 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[..., 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_3.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_deconv1 = self.deconv1(x_rec)
        x_deconv1 = torch.cat([x_deconv1, x_2], dim=1)
        x_deconv2 = self.deconv2(x_deconv1)
        x_deconv2 = torch.cat([x_deconv2, x_1], dim=1)
        x_deconv3 = self.deconv3(x_deconv2)
        x_out = self.conv(x_deconv3)
        return x_rot, x_contrastive, x_out


# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F


class Contrast(torch.nn.Module):
    def __init__(self, args, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = args.b
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(args.b * 2, args.b * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args).cuda()
        self.alpha_rot = 1.0
        self.alpha_contrast = 0.7
        self.alpha_recon = 3

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha_rot * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha_contrast * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha_recon * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)


class My_loss(Loss):
    def __init__(self, args):
        super(My_loss, self).__init__(args)

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        total_loss = rot_loss + contrast_loss

        return total_loss, (rot_loss, contrast_loss)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Training")
    args = parser.parse_args()
    args.spatial_dims = 3
    args.in_channels = 1
    args.feature_size = 24
    args.dropout_path_rate = 0.2
    args.use_checkpoint = False
    data_1 = torch.randn(2, 1, 25, 25, 25)
    data_2 = torch.zeros(2, 1, 24, 24, 24)
    net = My_SSLHead(args, dim=196, net_name='CNN')
    net.train()
    with torch.no_grad():
        out = net.forward(data_2)
    print()
