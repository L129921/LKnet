import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
# from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# class LSKblock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.bn0 = nn.BatchNorm2d(dim)
#         self.SiLU0 = nn.SiLU()
#
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.bn_spatial = nn.BatchNorm2d(dim)
#         self.SiLU_spatial = nn.SiLU()
#
#         self.conv1 = nn.Conv2d(dim, dim // 2, 1)
#         self.bn1 = nn.BatchNorm2d(dim // 2)
#         self.conv2 = nn.Conv2d(dim, dim // 2, 1)
#         self.bn2 = nn.BatchNorm2d(dim // 2)
#
#         self.ca = ChannelAttention(dim)
#         self.sa = SpatialAttention()
#
#     def forward(self, x):
#         out1 = self.conv0(x)
#         out1 = self.SiLU0(self.bn0(out1))
#
#         out2 = self.conv_spatial(out1)
#         out2 = self.SiLU_spatial(self.bn_spatial(out2))
#
#         out1 = self.conv1(out1)
#         out1 = self.bn1(out1)
#
#         out2 = self.conv2(out2)
#         out2 = self.bn2(out2)
#
#         attn = torch.cat([out1, out2], dim=1)
#         attn = self.ca(attn) * attn
#         attn = self.sa(attn) * attn
#
#         return attn


# class LSKblock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.bn0 = nn.BatchNorm2d(dim)
#         self.SiLU0 = nn.SiLU()
#
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.bn_spatial = nn.BatchNorm2d(dim)
#         self.SiLU_spatial = nn.SiLU()
#
#         self.conv1 = nn.Conv2d(dim, dim // 2, 1)
#         self.bn1 = nn.BatchNorm2d(dim // 2)
#         self.conv2 = nn.Conv2d(dim, dim // 2, 1)
#         self.bn2 = nn.BatchNorm2d(dim // 2)
#
#     def forward(self, x):
#         out1 = self.conv0(x)
#
#         out1 = self.SiLU0(self.bn0(out1))
#
#         out2 = self.conv_spatial(out1)
#         out2 = self.SiLU_spatial(self.bn_spatial(out2))
#
#         out1 = self.conv1(out1)
#         out1 = self.bn1(out1)
#
#         out2 = self.conv2(out2)
#         out2 = self.bn2(out2)
#
#         attn = torch.cat([out1, out2], dim=1)
#
#         return attn


#
#
# class LSKblock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.bn0 = nn.BatchNorm2d(dim)
#         self.SiLU0 = nn.SiLU()
#
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.bn_spatial = nn.BatchNorm2d(dim)
#         self.SiLU_spatial = nn.SiLU()
#
#         r = max(int(dim / 2), 32)
#         self.fc1 = nn.Linear(dim, r)
#         self.fcs = nn.ModuleList([])
#         for i in range(2):
#             self.fcs.append(
#                 nn.Linear(r, dim)
#             )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         out1 = self.conv0(x)
#         out1 = self.SiLU0(self.bn0(out1))
#
#         out2 = self.conv_spatial(x)
#         out2 = self.SiLU_spatial(self.bn_spatial(out2))
#
#         fu = out1 + out2
#         return fu



class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.bn0 = nn.BatchNorm2d(dim)
        self.SiLU0 = nn.SiLU()

        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.bn_spatial = nn.BatchNorm2d(dim)
        self.SiLU_spatial = nn.SiLU()

        r = max(int(dim / 2), 32)
        self.fc1 = nn.Linear(dim, r)

        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Linear(r, dim)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out1 = self.conv0(x)
        out1 = self.SiLU0(self.bn0(out1))

        out2 = self.conv_spatial(x)
        out2 = self.SiLU_spatial(self.bn_spatial(out2))
        feas = torch.cat([out1.unsqueeze_(dim=1), out2.unsqueeze_(dim=1)], dim=1)
        fu = torch.sum(feas, dim=1)
        fea_s = fu.mean(-1).mean(-1)
        fea_z = self.fc1(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        out = (feas * attention_vectors).sum(dim=1)

        return out

class Baseblock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.act1 = nn.SiLU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.act2 = nn.SiLU()

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = self.bn2(x)
        x = x + shorcut
        x = self.act2(x)
        return x


# class Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
#         super().__init__()
#         if norm_cfg:
#             self.norm1 = build_norm_layer(norm_cfg, dim)[1]
#             self.norm2 = build_norm_layer(norm_cfg, dim)[1]
#         else:
#             self.norm1 = nn.BatchNorm2d(dim)
#             self.norm2 = nn.BatchNorm2d(dim)
#         self.block = Baseblock(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#         layer_scale_init_value = 1e-2
#         self.layer_scale_1 = nn.Parameter(
#             layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
#
#         return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class LSKNet(BaseModule):
    def __init__(self, img_size=512, in_chans=3, embed_dims=[64, 128, 256, 512]
                 , drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 4, 2], num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        for i in range(num_stages):
            # 下采样
            patch_embed = OverlapPatchEmbed(img_size=img_size // (2 ** (i)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=2 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Baseblock(embed_dims[i]) for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LSKNet, self).init_weights()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

# imgs = np.zeros([1,3,224,224], dtype=np.float32)
# imgs = torch.tensor(imgs)
# model = LSKNet()
# output = model(imgs)
# a = 1
