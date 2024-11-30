import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

import models.vit_dino
from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

from . import tps_warp
from .tps_warp import _get_regular_grid
from .tps_warp import _generate_random_vectors
from .tps_warp import _thin_plate_spline_warp

import matplotlib.pylab as plt

import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from models import vit
from models import vit_dino

import logging
import mmcv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F
def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }

class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_para(self):
        self.brightness_para = random.uniform(-self.brightness_delta, self.brightness_delta)
        self.contrast_para = random.uniform(self.contrast_lower, self.contrast_upper)
        self.saturation_para = random.uniform(self.saturation_lower, self.saturation_upper)
        self.hue_para = random.randint(-self.hue_delta, self.hue_delta)

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.uniform(0, 1) < 0.5:
            return self.convert(
                img,
                beta=self.brightness_para)
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.uniform(0, 1) < 0.5:
            return self.convert(
                img,
                alpha=self.contrast_para)
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.uniform(0, 1) < 0.5:
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=self.saturation_para)
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.uniform(0, 1) < 0.5:
            img = mmcv.bgr2hsv(img)
            img[:, :,0] = (img[:, :, 0].astype(int) +self.hue_para) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, sample):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        # image_previous, image = sample['image_previous'], sample['image']

        self.get_para()

        # random brightness
        sample = self.brightness(sample)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.uniform(0, 1)
        if mode < 0.5:
            sample = self.contrast(sample)

        # random saturation
        sample = self.saturation(sample)

        # random hue
        sample = self.hue(sample)

        # random contrast
        if mode > 0.5:
            sample = self.contrast(sample)

        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(512, 512)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        # image = sample
        # h, w, _ = image.shape
        # print('image shape', image.shape)
        # print('gt shape', gt.shape)
        h, w = gt.shape

        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_CUBIC)

        left_top = (np.random.randint(0, h - self.output_size[0] + 1), np.random.randint(0, w - self.output_size[1] + 1))

        image_crop = image[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1], :]
        gt_crop = gt[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]

        sample.update({'image': image_crop, 'gt': gt_crop})

        return sample

class RandomCrop_t(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(512, 512)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        # image, gt = sample['image'], sample['gt']
        image = sample
        h, w, _ = image.shape

        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_CUBIC)

        left_top = (np.random.randint(0, h - self.output_size[0] + 1), np.random.randint(0, w - self.output_size[1] + 1))

        image_crop = image[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1], :]
        # gt_crop = gt[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]

        # sample.update({'image': image_crop, 'gt': gt_crop})

        return image_crop

class RandomResize(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, sample):
        h, w = random.randint(self.min, self.max), random.randint(self.min, self.max)

        # sample['image'] = cv2.resize(sample['image'], (w, h), interpolation=cv2.INTER_CUBIC)
        # sample['gt'] = cv2.resize(sample['gt'], (w, h), interpolation=cv2.INTER_NEAREST)
        sample = cv2.resize(sample, (w, h), interpolation=cv2.INTER_CUBIC)

        return sample

class RandomResize_t(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, sample):
        h, w = random.randint(self.min, self.max), random.randint(self.min, self.max)

        sample['image'] = cv2.resize(sample['image'], (w, h), interpolation=cv2.INTER_CUBIC)
        sample['gt'] = cv2.resize(sample['gt'], (w, h), interpolation=cv2.INTER_NEAREST)

        return sample

@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None, aug_type=None, batch=4):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.aug_type = aug_type
        self.batch = batch

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, gt = self.dataset[idx]

        # img_aug = image

        pmd_aug = PhotoMetricDistortion()
        # resize = RandomResize(int(self.inp_size * 1.05), int(self.inp_size * 1.2))
        resize = RandomResize_t(int(self.inp_size * 1.05), int(self.inp_size * 1.2))
        crop = RandomCrop((self.inp_size, self.inp_size))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

            sample = {'image': np.array(image), 'gt': np.array(gt)}

            sample = resize(sample)
            sample = crop(sample)
            image = Image.fromarray(sample['image'])
            gt = Image.fromarray(sample['gt'])

            # print('image shape',image.size)
            # print('gt shape', gt.size)



        return {
            'inp': self.img_transform(image),
            # 'aug': self.img_transform(img_aug),
            'gt': self.mask_transform(gt)
        }

