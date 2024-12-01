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
import random
from torch.nn import init
import thop

from .swin_transformer import swin_tiny, swin_small, swin_base, swin_base_384
from .mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .hrnet import HighResolutionNet
from .resnet import ResNet, Bottleneck

from torchvision import transforms


class Baseline(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder, feature_channels, embedding_dim = build_backbone(backbone)
        self.decoder = SegformerHead(feature_channels, embedding_dim)

    def forward(self, x):
        z = self.decoder(self.encoder(x))

        return z

def build_backbone(backbone):
    if backbone == 'mit_b1':
        b = mit_b1()
        saved_state_dict = torch.load('./mit_b1.pth')
        b.load_state_dict(saved_state_dict, strict=False)
        feature_channels = [64, 128, 320, 512]
        embedding_dim = 256
    elif backbone == 'swin_tiny':
        b = swin_tiny()
        saved_state_dict = torch.load('./swin_tiny_patch4_window7_224_22k.pth')
        b.load_state_dict(saved_state_dict['model'], strict=False)
        feature_channels = [96, 192, 384, 768]
        embedding_dim = 256
    elif backbone == 'resnet_50':
        b = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], replace_stride_with_dilation=[False, True, True])
        b.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
        feature_channels = [256, 512, 1024, 2048]
        embedding_dim = 256
    elif backbone == 'hrnet_18':
        b = HighResolutionNet('18')
        b.init_weights('./HRNet_W18_C_pretrained.pth')
        feature_channels = [18, 36, 72, 144]
        embedding_dim = 256


    return b, feature_channels, embedding_dim


import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F
import attr
from mmcv.cnn import ConvModule


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class SegformerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, embedding_dim, out_chan=1):
        super(SegformerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        embedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, out_chan, kernel_size=1)


    def forward(self, x):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # x = self.dropout(_c)
        self.decoder_feat = _c
        x = self.linear_pred(_c)

        return x
