import torch
import torch.nn as nn
import torch.nn.functional as F

import models
# import train
from models import register
from mmcv.runner import build_runner
import math
import numpy as np
from torch.autograd import Variable
import mmcv
from .mmseg.models import build_segmentor
from mmseg.models import backbones
from mmseg.models.builder import BACKBONES, SEGMENTORS
import os
import logging
logger = logging.getLogger(__name__)
from .iou_loss import IOU
import random
from torch.nn import init
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import thop
from .vit_mae import vit_base_patch16

from .vit_dino import vit_tiny, vit_base, vit_small, VisionTransformer
from .vit_mae import VisionTransformer as ViT_MAE, interpolate_pos_embed
from .vit_cls import VisionTransformer as ViT_CLS
from . import vit_dino as vits

from .vit_model import PUPHead, SegmenterConvHead
import sys
sys.path.append("..")
from train import *
from datasets import wrappers

#use bilateral solver
from misc import (
    batch_apply_bilateral_solver,
    set_seed,
    load_config,
)

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class VITDINO(nn.Module):
    def __init__(self, backbone='small'):
        super().__init__()
        # self.vit = vit_base(patch_size=16)
        # state_dict = torch.load('./dino_vitbase16_pretrain.pth')
        self.vit = vit_small(patch_size=8)
        state_dict = torch.load('./dino_deitsmall8_pretrain.pth')
        msg = self.vit.load_state_dict(state_dict, strict=False)
        self.embed_dim = self.vit.embed_dim
        self.decoder = PUPHead(self.embed_dim, hiddden_dim=512, depth=4, out_chan=1)

    def forward(self, x):
        img_feat = self.vit(x)
        x = self.decoder(img_feat)

        return x


class VITMAE(nn.Module):
    def __init__(self, backbone='small'):
        super().__init__()
        self.vit = ViT_MAE(img_size=224)
        self.vit.initialize_weights()
        self.embed_dim = self.vit.embed_dim
        self.decoder = PUPHead(self.embed_dim, hiddden_dim=512, depth=4, out_chan=1)

    def forward(self, x):
        img_feat = self.vit(x)
        x = self.decoder(img_feat)
        return x

class VITCLS(nn.Module):
    def __init__(self, backbone='base'):
        super().__init__()
        self.vit = ViT_CLS(model_name='vit_base_patch16_224', img_size=224, patch_size=16,
                           pos_embed_interp=True, drop_rate=0.,
                           embed_dim=768, depth=12, num_heads=12)
        self.embed_dim = self.vit.embed_dim

        self.decoder = PUPHead(self.embed_dim, hiddden_dim=512, depth=4, out_chan=1)
    def forward(self, x):
        img_feat = self.vit(x)
        x = self.decoder(img_feat)
        return x

from .archs import Baseline
from torchvision import transforms
@register('vit')
class VIT(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None, ag=False, part=None, cross=False, loss_type=None, w_bce=0, w_new=0, threshold=0, use_bs=None
):
        super(VIT, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inp_size = 400
        self.loss_mode = loss
        self.ag = ag
        self.cross = cross
        self.w_bce = w_bce
        self.w_new = w_new
        self.threshold = threshold
        self.use_bs = use_bs

        self.encoder = Baseline(encoder_mode['name'])



        model_total_params = sum(p.numel() for p in self.encoder.parameters())
        model_grad_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


        print('model_grad_params:' + str(model_grad_params),
              '\nmodel_total_params:' + str(model_total_params))


    def set_input(self, input, gt_mask, img_aug=None, epochs=0):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        # if self.ag == True:
        #     self.img_aug = img_aug.to(self.device)
        # else:
        #     self.img_aug = input.to(self.device)

        self.epoch = epochs

    def forward(self):

        #self.pred_mask = self.encoder(self.img_aug)
        self.pred_mask = self.encoder(self.input)
        self.pred_mask = F.interpolate(self.pred_mask, size=self.input.shape[2:], mode='bilinear', align_corners=False)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.gt_mask = F.interpolate(self.gt_mask, size=self.pred_mask.shape[2:], mode='bilinear', align_corners=False)

        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask) * self.w_bce

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()
            self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask) * self.w_bce


        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad