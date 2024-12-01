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
from datasets.wrappers import PhotoMetricDistortion, RandomResize, RandomCrop, RandomCrop_t
import torch.nn.functional as F
from tqdm import tqdm
from models.vit import BBCEWithLogitLoss

#tent: fully test-time adaptation
import tent
import math

#save log file
import sys, time

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x = x.view(x.size(0), -1)
    entropy = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return entropy

#moving average technique
def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

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
            with open(self.filename, 'w') as log:
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
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # writing the log file
    sys.stdout = Logger(sys.stdout)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # save_name = args.name

    # if save_name is None:
    #     save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    # save_path = os.path.join('./save', save_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(config['model']).cuda()
    model.encoder.load_state_dict(torch.load(args.model))
    # model.encoder.load_state_dict(torch.load(args.model), strict=False)

    # python ttt_new.py --model ./mit_b4.pth --resolution 352,352 --gpu 0 --config configs/demo.yaml
    h, w = list(map(int, args.resolution.split(',')))
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

    #shadow_istd
    # img_dir = './load/ISTD_Dataset/test/test_A'
    # save_dir = '/Ablation2/save/result_ISTD-test'
    # ground_truth_dir = './load/ISTD_Dataset/test/test_B'

    img_dir = config['val_dataset']['dataset']['args']['root_path_1']
    # save_dir = '/Ablation2/save/result_ISTD-test'
    ground_truth_dir = config['val_dataset']['dataset']['args']['root_path_2']

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
    # crop_aug = RandomCrop((w, h))
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
    epochs = 5 + 1
    # Record start time
    start_time = time.time()

    for epoch in range(epochs):
        metric1[epoch], metric2[epoch], metric3[epoch], metric4[epoch] = [], [], [], []
    learning_rate = 1e-5 #5e-7 
    # print(len(img_filenames))
    for i in range(len(img_filenames)):
        print(str(i) + ' / ' + str(len(img_filenames)))
        img_path = os.path.join(img_dir, img_filenames[i])
        gt_path = os.path.join(ground_truth_dir, gt_filenames[i])

        img = Image.open(img_path).convert('RGB')
        g_t = Image.open(gt_path).convert('L')

        # eval pretrained model
        image = img_transform(img)
        gt = mask_transform(g_t)

        image = image.cuda()

        # start test time training

        # initialize pretrained model
        model.encoder.load_state_dict(torch.load(args.model)) #make sure the inference for each sample is independent
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        model.eval()
        with torch.no_grad():
            pred = model.encoder(image.unsqueeze(0))
            pred = F.interpolate(pred, size=image.shape[1:], mode='bilinear', align_corners=False)
            pred = torch.sigmoid(pred)

        m1, m2, m3, m4 = metric_fn(pred.cpu(), gt.unsqueeze(0)) #evaluate the pretrained(base) model
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

        #strat tent training
        for epoch in range(1, epochs):
            batch_size = 4
            image_aug = []
            #data augmentation
            for b in range(batch_size):
                image_a = img
                if random.random() < 0.5:
                    image_a = image_a.transpose(Image.FLIP_LEFT_RIGHT)
                image_a = resize_aug(np.array(image_a))
                image_a = crop_aug(image_a)
                image_aug.append(Image.fromarray(np.uint8(image_a)))
            #stack augmented data into batch_size
            i = []
            for b in range(batch_size):
                i.append(img_transform(image_aug[b]).unsqueeze(0).cuda())
            image_aug = torch.cat(i, dim=0)

            #train tent
            model.train()
            # tent_model = tent.Tent(model=model, optimizer=optimizer, steps=1, episodic=False)

            #forward
            outputs = model.encoder(image_aug)
            outputs = F.interpolate(outputs, size=image.shape[1:], mode='bilinear', align_corners=False)

            #entropy loss calculation
            entropys = softmax_entropy(outputs)  

            e_margin = 6.5
            d_margin = 0.05
            filter_ids_1 = torch.where(entropys < e_margin)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0] > -0.1) 
            entropys = entropys[filter_ids_1]

            current_model_probs = None

            # filter redundant samples
            if current_model_probs is not None:
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0),
                                                          outputs[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
         
            coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples

            loss = entropys.mean(0)

            # adapt
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # eval test time trained model
            model.eval()
            with torch.no_grad():
                pred = model.encoder(image.unsqueeze(0))
                pred = F.interpolate(pred, size=image.shape[1:], mode='bilinear', align_corners=False)
                pred = torch.sigmoid(pred) #1582

            m1, m2, m3, m4 = metric_fn(pred.cpu(), gt.unsqueeze(0)) #evaluate the epoch th test-time trained model

            metric1[epoch].append(m1)
            metric2[epoch].append(m2)
            metric3[epoch].append(m3)
            metric4[epoch].append(m4)


    for epoch in range(epochs):
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
