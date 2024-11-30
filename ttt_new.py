import argparse
import os
from PIL import Image

import torch
from torchvision import transforms
import time
import models
from utils import make_coord
from test import batched_predict
import yaml
# from utils import to_pixel_samples
import cv2
import numpy as np
from torchsummary import  summary
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import utils
import shutil
import random
# from datasets import wrappers
from datasets.wrappers import PhotoMetricDistortion, RandomResize, RandomCrop_t
import torch.nn.functional as F
from tqdm import tqdm
from models.vit import BBCEWithLogitLoss

#save log file
import sys, time

class Logger(object):
    def __init__(self, stream=sys.stdout, add_flag=True):
        self.terminal = stream
        self.name = args.name
        output_dir = './save'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = self.name + '.txt'
        self.filename = os.path.join(output_dir, log_name)
        self.add_flag = add_flag

        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            # with open(self.filename, 'w') as log:
            with open(self.filename, 'a+' if self.add_flag else 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--config', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--loss', default=None)
    parser.add_argument('--eval_type', default=None)
    parser.add_argument('--img_dir', default=None)
    parser.add_argument('--gt_dir', default=None)
    parser.add_argument('--lr', default=None)
    parser.add_argument('--fg_cons', default=None)
    parser.add_argument('--bg_cons', default=None)
    parser.add_argument('--w_fg', default=None)
    parser.add_argument('--w_bg', default=None)
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # writing the log file
    sys.stdout = Logger(sys.stdout)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(config['model']).cuda()
    model.encoder.load_state_dict(torch.load(args.model))

    # python ttt_new.py --model ./mit_b4.pth --resolution 352,352 --gpu 0 --config configs/demo.yaml
    # h, w = list(map(int, args.resolution.split(',')))
    h = w = config['model']['args']['inp_size']
    img_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
    ])

    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1, 1, 1])
    ])

    # img_dir = './load/ISTD_Dataset/test/test_A'
    # save_dir = '/Ablation2/save/result_ISTD-test'
    # ground_truth_dir = './load/ISTD_Dataset/test/test_B'

    img_dir = config['val_dataset']['dataset']['args']['root_path_1']
    save_dir = '/Ablation2/save/result_ISTD-test'
    ground_truth_dir = config['val_dataset']['dataset']['args']['root_path_2']

    # img_dir = args.img_dir
    # save_dir = '/Ablation2/save/result_ISTD-test'
    # ground_truth_dir = args.gt_dir

    eval_type = config['eval_type']
    if eval_type == 'sod':
        # metric_fn = utils.calc_uda
        metric_fn = utils.calc_sod
        metric1_name, metric2_name, metric3_name, metric4_name = 'f_max', 'mae', 's_max', 'e_max'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1_name, metric2_name, metric3_name, metric4_name = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1_name, metric2_name, metric3_name, metric4_name = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1_name, metric2_name, metric3_name, metric4_name = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1_name, metric2_name, metric3_name, metric4_name = 'sm', 'em', 'wfm', 'mae'
    
    img_filenames = sorted(os.listdir(img_dir))
    gt_filenames = sorted(os.listdir(ground_truth_dir))

    pmd_aug = PhotoMetricDistortion()
    # resize_aug = RandomResize(int(w * 1.05), int(h * 1.5))
    resize_aug = RandomResize(int(w * 1.05), int(h * 1.2))
    crop_aug = RandomCrop_t((w, h))

    # control trainable parameters
    for k, p in model.encoder.named_parameters():
        if 'decode' in k:
            p.requires_grad = False
        else:
            p.requires_grad = True

    parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    model_total_params = sum(p.numel() for p in model.encoder.parameters())
    model_grad_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params),
          '\nmodel_total_params:' + str(model_total_params))

    metric1, metric2, metric3, metric4 = {}, {}, {}, {}
    epochs = 20 + 1
    # Record start time
    start_time = time.time()

    for epoch in range(epochs):
        metric1[epoch], metric2[epoch], metric3[epoch], metric4[epoch] = [], [], [], []
        print('epoch1', epoch)
    learning_rate = 5e-7 #istd: 5e-5; sbu:5e-7
    # print(len(img_filenames))
    for i in range(len(img_filenames)):
        print(str(i) + ' / ' + str(len(img_filenames)))
        img_path = os.path.join(img_dir, img_filenames[i])
        gt_path = os.path.join(ground_truth_dir, gt_filenames[i])

        image = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')

        # eval pretrained model
        input = img_transform(image)
        gt = mask_transform(gt)

        input = input.cuda()

        # start test time training

        # initialize pretrained model
        model.encoder.load_state_dict(torch.load(args.model))
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        model.eval()
        with torch.no_grad():
            pred = model.encoder(input.unsqueeze(0))
            pred = F.interpolate(pred, size=input.shape[1:], mode='bilinear', align_corners=False)
            pred = torch.sigmoid(pred)

        m1, m2, m3, m4 = metric_fn(pred.cpu(), gt.unsqueeze(0))
        metric1[0].append(m1)
        metric2[0].append(m2)
        metric3[0].append(m3)
        metric4[0].append(m4)

        log_info = ['epoch {}/{}'.format(epoch, epochs)]
        # log_info.append('test: loss={:.4f}'.format(loss))

        log_info.append(metric1_name.format(np.mean(metric1[0])))
        log_info.append(metric2_name.format(np.mean(metric2[0])))
        log_info.append(metric3_name.format(np.mean(metric3[0])))
        log_info.append(metric4_name.format(np.mean(metric4[0])))

        for epoch in range(1, epochs):
            print('epoch2', epoch)
            batch_size = 4

            sample_image = []
            sample_image_aug = []

            for b in range(batch_size):
                image_1, image_2 = image, image
                if random.random() < 0.5:
                    image_1 = image_1.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:
                    image_2 = image_2.transpose(Image.FLIP_LEFT_RIGHT)
                image_1 = resize_aug(np.array(image_1))
                image_2 = resize_aug(np.array(image_2))

                image_1 = crop_aug(image_1)
                image_2 = crop_aug(image_2)
                sample_image.append(np.array(image_1))
                sample_image_aug.append(np.array(image_2))

            i1, i2 = [], []
            for b in range(batch_size):
                i1.append(img_transform(Image.fromarray(sample_image[b])).unsqueeze(0).cuda())
                i2.append(img_transform(Image.fromarray(sample_image_aug[b])).unsqueeze(0).cuda())
            image_aug1 = torch.cat(i1, dim=0)
            image_aug2 = torch.cat(i2, dim=0)

            # test time training
            model.train()
            image_aug1 = F.interpolate(image_aug1, size=(w, h), mode='bilinear', align_corners=False)
            image_aug2 = F.interpolate(image_aug2, size=(w, h), mode='bilinear', align_corners=False)
            pred_aug1 = model.encoder(image_aug1)
            pred_aug2 = model.encoder(image_aug2)

            pred_aug1 = torch.sigmoid(
                F.interpolate(pred_aug1, size=image_aug1.shape[2:], mode='bilinear', align_corners=False))
            pred_aug2 = torch.sigmoid(
                F.interpolate(pred_aug2, size=image_aug2.shape[2:], mode='bilinear', align_corners=False))

            threshold = 0.5
            inter_fg = torch.logical_and(pred_aug1 > threshold, pred_aug2 > threshold)
            inter_bg = torch.logical_and(pred_aug1 < (1 - threshold), pred_aug2 < (1 - threshold))

            loss = 0
            mae = torch.nn.L1Loss()
            mse = torch.nn.MSELoss()

            print('args.fg_cons', args.fg_cons)
            if args.fg_cons == 'True':
                if True not in inter_fg:
                    loss += 0
                else:
                    # loss += F.l1_loss(pred_aug1[inter_fg], pred_aug2[inter_fg], reduction='mean')
                    p_fg, p_bg = torch.sigmoid(pred_aug1[inter_fg]).unsqueeze(0), 1 - torch.sigmoid(
                        pred_aug1[inter_fg]).unsqueeze(0)
                    pp_fg, pp_bg = torch.sigmoid(pred_aug2[inter_fg]).unsqueeze(0), 1 - torch.sigmoid(
                        pred_aug2[inter_fg]).unsqueeze(0)
                    p, pp = torch.cat([p_fg, p_bg], dim=0), torch.cat([pp_fg, pp_bg], dim=0)
                    loss += 0.5 * F.kl_div(torch.log(p), pp, reduction='batchmean')
            else:
                loss += 0

            print('args.bg_cons', args.bg_cons)
            if args.bg_cons == 'True':
                if True not in inter_bg:
                    loss += 0
                else:
                    p_fg, p_bg = torch.sigmoid(pred_aug1[inter_bg]).unsqueeze(0), 1 - torch.sigmoid(
                        pred_aug1[inter_bg]).unsqueeze(0)
                    pp_fg, pp_bg = torch.sigmoid(pred_aug2[inter_bg]).unsqueeze(0), 1 - torch.sigmoid(
                        pred_aug2[inter_bg]).unsqueeze(0)
                    p, pp = torch.cat([p_fg, p_bg], dim=0), torch.cat([pp_fg, pp_bg], dim=0)
                    loss += F.kl_div(torch.log(p), pp, reduction='batchmean')
                    # loss += F.l1_loss(pred_aug1[inter_bg], pred_aug2[inter_bg], reduction='mean')
            else:
                loss += 0

            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # eval test time trained model
            model.eval()
            with torch.no_grad():
                pred = model.encoder(input.unsqueeze(0))
                pred = F.interpolate(pred, size=input.shape[1:], mode='bilinear', align_corners=False)
                pred = torch.sigmoid(pred)

            m1, m2, m3, m4 = metric_fn(pred.cpu(), gt.unsqueeze(0))

            metric1[epoch].append(m1)
            metric2[epoch].append(m2)
            metric3[epoch].append(m3)
            metric4[epoch].append(m4)


    for epoch in range(epochs):
        print('epoch3', epoch)
        log_info = ['epoch {}/{}'.format(epoch, epochs)]
        # log_info.append('test: loss={:.4f}'.format(loss))

        # ignore nan
        log_info.append(metric1_name.format(np.nanmean(metric1[epoch])))
        log_info.append(metric2_name.format(np.nanmean(metric2[epoch])))
        log_info.append(metric3_name.format(np.nanmean(metric3[epoch])))
        log_info.append(metric4_name.format(np.nanmean(metric4[epoch])))

        print('epoch', epoch, '/', epochs-1, metric1_name, np.nanmean(metric1[epoch]), metric2_name, np.nanmean(metric2[epoch]), metric3_name, np.nanmean(metric3[epoch]), metric4_name, np.nanmean(metric4[epoch]))

        # Calculate time cost
        end_time = time.time()
        time_cost = end_time - start_time
        hours = int(time_cost // 3600)
        minutes = int((time_cost % 3600) // 60)
        seconds = int(time_cost % 60)

        print("Time cost: {}h{}mins{}s".format(hours, minutes, seconds))
        # print("Time cost: {:.2f} seconds".format(time_cost))