import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples,to_pixel_samples_bias,make_coord


@register('sr-implicit-float-paired-test')
class SRImplicitFloatPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q


    def set_test_scale(self, evalscale):
        self.dataset.set_test_scale(evalscale)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        crop_lr, crop_hr = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register("sr-implicit-float-paired")
class SRImplicitFloatPaired(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def set_test_scale(self, evalscale):
        self.dataset.set_test_scale(evalscale)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        w, h = img_hr.shape[-2:]
        # print(img_lr.shape)

        s = img_hr.shape[-2] / img_lr.shape[-2]  # assume int scale
        # print(s)
        if self.inp_size is None:
            # h_lr, w_lr = img_lr.shape[-2:]
            # img_hr = img_hr[:, :int(h_lr * s), :int(w_lr * s)]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0 : x0 + w_lr, y0 : y0 + w_lr]
            w_hr = round(w_lr * s)
            x1 = round(x0 * s)
            y1 = round(y0 * s)
            crop_hr = img_hr[:, x1 : x1 + w_hr, y1 : y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            try:
                sample_lst = np.random.choice(
                    len(hr_coord), self.sample_q, replace=False
                )
                hr_coord = hr_coord[sample_lst]
                hr_rgb = hr_rgb[sample_lst]

            except:
                print(len(hr_coord), s)
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            "inp": crop_lr,
            "coord": hr_coord,
            "cell": cell,
            "gt": hr_rgb,
            "w": w,
            "h": h,
        }