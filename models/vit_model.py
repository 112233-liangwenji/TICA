import torch
import torch.nn as nn
import torch.nn.functional as F


import models
from models import register
import math
import numpy as np
from torch.autograd import Variable

from timm.models.layers import trunc_normal_
import os
import logging
logger = logging.getLogger(__name__)
from functools import partial
# from .adv_loss import AdversarialLoss
# import random
# from torch.nn import init
# import thop
# from .mae import get_2d_sincos_pos_embed
from timm.models.vision_transformer import Block
from .vit_dino import vit_tiny, vit_base, vit_small
from .vit_mae import VisionTransformer as ViT_MAE, interpolate_pos_embed
from .vit_cls import VisionTransformer as ViT_CLS


class Baseline(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone == 'dino':
            self.encoder = VITDINO()
        elif backbone == 'mae':
            self.encoder = VITMAE()
        elif backbone == 'cls':
            self.encoder = VITCLS()

        self.embed_dim = self.encoder.embed_dim

        self.decoder = PUPHead(self.embed_dim, hiddden_dim=512, depth=4, out_chan=1)

        print(sum(p.numel() for p in self.encoder.parameters()),
              sum(p.numel() for p in self.decoder.parameters()))

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        return x



class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=256, depth=4, num_heads=8):
        super(SelfAttentionBlock, self).__init__()
        self.dim = dim
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = 4.
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_rate = 0.1
        self.attn_drop_rate = 0.
        self.drop_path_rate = 0.

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(self.depth)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        return x


class PUPHead(nn.Module):
    def __init__(self, dim, hiddden_dim=512, depth=4, out_chan=1):
        super(PUPHead, self).__init__()
        self.dim = dim
        self.hiddden_dim = hiddden_dim
        self.depth = depth
        self.out_chan = out_chan

        # self.proj = nn.Linear(self.dim, self.hiddden_dim)
        # self.decoder = SelfAttentionBlock(dim=self.hiddden_dim, depth=depth)
        #
        # self.conv_0 = nn.Conv2d(
        #     hiddden_dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        # self.conv_1 = nn.Conv2d(
        #     hiddden_dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        # self.conv_2 = nn.Conv2d(
        #     hiddden_dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        # self.conv_3 = nn.Conv2d(
        #     hiddden_dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        # self.conv_4 = nn.Conv2d(hiddden_dim, self.out_chan, kernel_size=1, stride=1)

        self.conv_0 = nn.Conv2d(
            dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            hiddden_dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(
            hiddden_dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(
            hiddden_dim, hiddden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(hiddden_dim, self.out_chan, kernel_size=1, stride=1)


    def forward(self, x):

        # x = self.proj(x)
        #
        # x = self.decoder(x)

        b, c = x.shape[0], x.shape[2]
        h = w = int(x.shape[1] ** .5)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.conv_0(x)
        x = F.relu(x)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=False)
        x = self.conv_1(x)
        x = F.relu(x)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=False)
        x = self.conv_2(x)
        x = F.relu(x)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=False)
        x = self.conv_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode='bilinear', align_corners=False)
        return x

def up_block(
    in_dim,
    out_dim,
    block_depth,
    kernel_size=3,
    padding_mode="zeros",
    no_upsampling=False,
):
    uplayer = nn.Upsample(scale_factor=(2, 2), mode="nearest")
    layers = [
        nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
        ),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(),
    ]
    for i in range(block_depth - 1):
        layers += [
            nn.Conv2d(
                out_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(),
        ]
    if not no_upsampling:
        # layers.insert(0, uplayer)
        layers.insert(0, nn.Upsample(scale_factor=2, mode="nearest"))
    return nn.Sequential(*layers)

class SegmenterConvHead(nn.Module):
    def __init__(
        self, upsampling_blocks, in_channels, widths, mask_channels, block_depth
    ):
        super().__init__()
        assert len(widths) == upsampling_blocks + 1
        layers = []
        widths = [in_channels] + widths

        for i in range(len(widths) - 1):
            layers.append(
                up_block(
                    widths[i],
                    widths[i + 1],
                    block_depth,
                    no_upsampling=(i == len(widths) - 2),
                )
            )
        layers.append(
            nn.Conv2d(widths[-1], mask_channels, kernel_size=1, stride=1, padding=0)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, return_log_imgs=False):
        x = self.model(x)
        # TODO: probably not needed
        if return_log_imgs:
            return x, {}
        return x

class VITDINO(nn.Module):
    def __init__(self, backbone='small'):
        super().__init__()
        self.vit = vit_base(patch_size=16)
        state_dict = torch.load('./dino_vitbase16_pretrain.pth')
        msg = self.vit.load_state_dict(state_dict, strict=False)
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        img_feat = self.vit(x)

        return img_feat


class VITMAE(nn.Module):
    def __init__(self, backbone='small'):
        super().__init__()
        self.vit = ViT_MAE(img_size=352)
        self.vit.initialize_weights()
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        img_feat = self.vit(x)

        return img_feat

class VITCLS(nn.Module):
    def __init__(self, backbone='base'):
        super().__init__()
        self.vit = ViT_CLS(model_name='vit_base_patch16_224', img_size=352,
                                   pos_embed_interp=True, drop_rate=0.,
                                   embed_dim=768, depth=12, num_heads=12)
        self.embed_dim = self.vit.embed_dim


    def forward(self, x):
        img_feat = self.vit(x)

        return img_feat