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
import torch.nn as nn

#save log file
import sys, time

#bs
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import cg

#bilateral solver post-processing
RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)
MAX_VAL = 255.0

def rgb2yuv(im):
    # im = im.transpose(1, 2, 0)[:, :, 0]
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET

def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs

class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        print('im shape', im.shape)
        im = im.transpose(1, 2, 0)
        print('im.shape', im.shape)
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] / sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL ** np.arange(self.dim))
        # Construct S and B matrix
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = \
            np.unique(hashed_coords, return_index=True, return_inverse=True)
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                          (valid_coord, idx)),
                                         shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)

    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)

    def slice(self, y):
        return self.S.T.dot(y)

    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) / \
            self.slice(self.blur(self.splat(np.ones_like(x))))

def bistochastize(grid, maxiter=10):
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in range(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)
    Dm = diags(m, 0)
    Dn = diags(n, 0)
    return Dn, Dm

class BilateralSolver(object):
    def __init__(self, grid, params):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)

    def solve(self, x, w):
        # Check that w is a vector or a nx1 matrix
        if w.ndim == 2:
            assert (w.shape[1] == 1)
        elif w.dim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:, 0], 0)
        A = self.params["lam"] * A_smooth + A_data
        xw = x * w
        b = self.grid.splat(xw)
        # Use simple Jacobi preconditioner
        A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
        M = diags(1 / A_diag, 0)
        # Flat initialization
        y0 = self.grid.splat(xw) / w_splat
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], info = cg(A, b[..., d], x0=y0[..., d], M=M, maxiter=self.params["cg_maxiter"],
                                    tol=self.params["cg_tol"])
        xhat = self.grid.slice(yhat)
        return xhat

def bilateral_solver_output(image, target, sigma_spatial=24, sigma_luma=4, sigma_chroma=4): #sigma_spatial=24 sigma_luma=4, sigma_chroma=4 2100; sigma_spatial=8, sigma_luma=4, sigma_chroma=4 2101, 2106; sigma_spatial=16, sigma_luma=16, sigma_chroma=8 2102-2104, 2107
    image = image.cpu().numpy()
    # image = image.astype(np.uint8)
    reference = np.array(image)
    # h, w = target.shape
    h = target.shape[2]
    w = target.shape[3]
    confidence = np.ones((h, w)) * 0.999

    grid_params = {
        'sigma_luma': 2, #sigma_luma,  # Brightness bandwidth
        'sigma_chroma': 2, #sigma_chroma,  # Color bandwidth
        'sigma_spatial': 2 #sigma_spatial  # Spatial bandwidth
    }

    bs_params = {
        'lam': 128,  # The strength of the smoothness parameter 256/2100; 128/2101; 256/2102-2104; 128/2106;
        'A_diag_min': 1e-5,  # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
        'cg_tol': 1e-5,  # The tolerance on the convergence in PCG
        'cg_maxiter': 25  # The number of PCG iterations
    }

    grid = BilateralGrid(reference, **grid_params)

    #conversion
    target = target.cpu().numpy()

    t = target.reshape(-1, 1).astype(np.double)
    c = confidence.reshape(-1, 1).astype(np.double)

    ## output solver, which is a soft value
    output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape((h, w))

    binary_solver = ndimage.binary_fill_holes(output_solver > 0.5)
    labeled, nr_objects = ndimage.label(binary_solver)

    nb_pixel = [np.sum(labeled == i) for i in range(nr_objects + 1)]
    pixel_order = np.argsort(nb_pixel)
    try:
        binary_solver = labeled == pixel_order[-2]
    except:
        binary_solver = np.ones((h, w), dtype=bool)

    return output_solver, binary_solver

def IoU(mask1, mask2):
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()


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

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(config['model']).cuda()
    model.encoder.load_state_dict(torch.load(args.model))

    # python ttt_new.py --model ./mit_b4.pth --resolution 352,352 --gpu 0 --config configs/demo.yaml
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

    img_dir = config['val_dataset']['dataset']['args']['root_path_1']
    ground_truth_dir = config['val_dataset']['dataset']['args']['root_path_2']

    eval_type = config['eval_type']
    if eval_type == 'sod':
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
    epochs =1 + 1
    # Record start time
    start_time = time.time()

    for epoch in range(epochs):
        metric1[epoch], metric2[epoch], metric3[epoch], metric4[epoch] = [], [], [], []
        # print('epoch1', epoch)
    learning_rate = 1e-5
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

        # initialize pretrained model
        model.encoder.load_state_dict(torch.load(args.model)) #make sure the inference for each sample is independent
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        model.eval()
        with torch.no_grad():
            pred = model.encoder(image.unsqueeze(0))
            pred = F.interpolate(pred, size=image.shape[1:], mode='bilinear', align_corners=False)
            pred = torch.sigmoid(pred)
            # print('pred', pred.shape) # torch.Size([1, 1, 400, 400])

            #start bs post-processing
            # image = image.cpu().numpy()
            # post_pred = dense_crf(image, pred)
            _ , binary_solver = bilateral_solver_output(image, pred, sigma_spatial=16, sigma_luma=16, sigma_chroma=8)

            # mask1 = torch.from_numpy(pred).cuda()
            mask1 = pred.squeeze()
            # print('mask1', mask1.shape) #torch.Size([400, 400])
            mask2 = torch.from_numpy(binary_solver).cuda()

            if IoU(mask1, mask2) < 0.5:
                binary_solver = binary_solver * -1

            # post_pred process
            post_pred = torch.from_numpy(binary_solver)
            # post_pred = binary_solver
            # print('before post_pred', post_pred.shape)  # torch.Size([400, 400])
            post_pred = post_pred.unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 400, 400])
            post_pred = post_pred.type(torch.float32)
            # print('after post_pred', post_pred.shape)
            # print('pred value', pred)
            # print('post_pred value', post_pred)

        #baseline model metric
        m1, m2, m3, m4 = metric_fn(pred.cpu(), gt.unsqueeze(0))
        metric1[0].append(m1)
        metric2[0].append(m2)
        metric3[0].append(m3)
        metric4[0].append(m4)

        #after crf metric
        m1, m2, m3, m4 = metric_fn(post_pred.cpu(), gt.unsqueeze(0))
        metric1[1].append(m1)
        metric2[1].append(m2)
        metric3[1].append(m3)
        metric4[1].append(m4)

        log_info = ['epoch {}/{}'.format(epoch, epochs)]
        # log_info.append('test: loss={:.4f}'.format(loss))

        log_info.append(metric1_name.format(np.mean(metric1[0])))
        log_info.append(metric2_name.format(np.mean(metric2[0])))
        log_info.append(metric3_name.format(np.mean(metric3[0])))
        log_info.append(metric4_name.format(np.mean(metric4[0])))


    for epoch in range(epochs):
        # print('epoch3', epoch)
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






