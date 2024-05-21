import os
import json
from PIL import Image

import pickle
import imageio
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
import cv2
import math


idx_with_6 = [154,155,156,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173
                ,174,175,176,177,178,179,180,181,182,183,184,186,187,188,189,190]

## code for real arbitrary SR dataset

class RealImageFolder(Dataset):

    def __init__(self, root_path,first_k=None,
                 repeat=1, cache='none', data_type='LR'):
        self.repeat = repeat
        self.cache = cache
        self.data_type = data_type

        self.idx_without_6 = []
        self.test_scale = 0

        if self.data_type == 'HR':
            self.files = []
            filenames = sorted(os.listdir(root_path))
            if first_k is not None:
                filenames = filenames[:first_k]

            for filename in filenames:
                file = os.path.join(root_path, filename)

                if cache == 'none':
                    self.files.append(file)

                elif cache == 'bin':
                    bin_root = os.path.join(os.path.dirname(root_path),
                        '_bin_' + os.path.basename(root_path))
                    if not os.path.exists(bin_root):
                        os.mkdir(bin_root)
                        print('mkdir', bin_root)
                    bin_file = os.path.join(
                        bin_root, filename.split('.')[0] + '.pkl')
                    if not os.path.exists(bin_file):
                        with open(bin_file, 'wb') as f:
                            pickle.dump(imageio.imread(file), f)
                        print('dump', bin_file)
                    self.files.append(bin_file)

                elif cache == 'in_memory':
                    self.files.append(transforms.ToTensor()(
                        Image.open(file).convert('RGB')))
        
        else:
            self.files = []
            filename_list = []
            dirnames = sorted(os.listdir(root_path))

            for dirname in dirnames:
                dir_path = os.path.join(root_path,dirname)
                file_list = sorted([os.path.join(dirname, filename) for filename in os.listdir(dir_path)])
                if file_list:
                    filename_list.append(file_list)                  
                
              
            if first_k is not None:
                filename_list = filename_list[:first_k]

            for filenames in filename_list:
                files = []
                # print(filenames)
                for filename in filenames:
                    file = os.path.join(root_path, filename)

                    if cache == 'none':
                        files.append(file)

                    elif cache == 'bin':
                        bin_root = os.path.join(os.path.dirname(root_path),
                            '_bin_' + os.path.basename(root_path))
                        if not os.path.exists(bin_root):
                            os.mkdir(bin_root)
                            print('mkdir', bin_root)
                        bin_file = os.path.join(
                            bin_root, filename.split('.')[0] + '.pkl')
                        if not os.path.exists(bin_file):
                            with open(bin_file, 'wb') as f:
                                pickle.dump(imageio.imread(file), f)
                            print('dump', bin_file)
                        files.append(bin_file)

                    elif cache == 'in_memory':
                        files.append(transforms.ToTensor()(
                            Image.open(file).convert('RGB')))

                self.files.append(files)                  
            

    def set_test_scale(self,test_scale):
        self.test_scale = test_scale


    def __len__(self):
        if self.test_scale == 6:
            return len(idx_with_6) *self.repeat
        
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        if self.test_scale == 6:
            x = self.files[idx_with_6[idx % len(idx_with_6)] - 154]
        else:
            x = self.files[idx % len(self.files)]
        # print(idx)

        if self.data_type =='LR':
            if self.test_scale ==0:
                x = random.choice(x)
            else:
                x = x[self.test_scale-2]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

@register('paired-real-image-folders')
class PairedRealImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = RealImageFolder(root_path_1, **kwargs)
        self.dataset_2 = RealImageFolder(root_path_2, **kwargs, data_type='HR')


    def set_test_scale(self,test_scale):
        self.dataset_1.set_test_scale(test_scale)
        self.dataset_2.set_test_scale(test_scale)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]





    
    

class RealImageFolderTest(Dataset):

    def __init__(self, root_path,first_k=None,
                 repeat=1, cache='none', data_type='LR'):
        self.repeat = repeat
        self.cache = cache
        self.data_type = data_type

        self.idx_without_6 = []
        self.test_scale = 0

        if self.data_type == 'HR':
            self.files = []
            filenames = sorted(os.listdir(root_path))
            if first_k is not None:
                filenames = filenames[:first_k]

            for filename in filenames:
                file = os.path.join(root_path, filename)

                if cache == 'none':
                    self.files.append(file)

                elif cache == 'bin':
                    bin_root = os.path.join(os.path.dirname(root_path),
                        '_bin_' + os.path.basename(root_path))
                    if not os.path.exists(bin_root):
                        os.mkdir(bin_root)
                        print('mkdir', bin_root)
                    bin_file = os.path.join(
                        bin_root, filename.split('.')[0] + '.pkl')
                    if not os.path.exists(bin_file):
                        with open(bin_file, 'wb') as f:
                            pickle.dump(imageio.imread(file), f)
                        print('dump', bin_file)
                    self.files.append(bin_file)

                elif cache == 'in_memory':
                    self.files.append(transforms.ToTensor()(
                        Image.open(file).convert('RGB')))
        
        else:
            self.files = []
            filename_list = []
            dirnames = sorted(os.listdir(root_path))

            for dirname in dirnames:
                dir_path = os.path.join(root_path,dirname)
                file_list = sorted([os.path.join(dirname, filename) for filename in os.listdir(dir_path)])
                if file_list:
                    filename_list.append(file_list)                  
                
              
            if first_k is not None:
                filename_list = filename_list[:first_k]

            for filenames in filename_list:
                files = []
                # print(filenames)
                for filename in filenames:
                    file = os.path.join(root_path, filename)

                    if cache == 'none':
                        # img = cv2.imread(file)
                        # cv2.imwrite(file, img)
                        files.append(file)

                    elif cache == 'bin':
                        bin_root = os.path.join(os.path.dirname(root_path),
                            '_bin_' + os.path.basename(root_path))
                        if not os.path.exists(bin_root):
                            os.mkdir(bin_root)
                            print('mkdir', bin_root)
                        bin_file = os.path.join(
                            bin_root, filename.split('.')[0] + '.pkl')
                        if not os.path.exists(bin_file):
                            with open(bin_file, 'wb') as f:
                                pickle.dump(imageio.imread(file), f)
                            print('dump', bin_file)
                        files.append(bin_file)

                    elif cache == 'in_memory':
                        files.append(transforms.ToTensor()(
                            Image.open(file).convert('RGB')))

                self.files.append(files)      
                
                            
            

    def set_test_scale(self,test_scale):
        self.test_scale = test_scale


    def __len__(self):
        if self.test_scale == 6:
            return len(idx_with_6) *self.repeat
        
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        if self.test_scale == 6:
            x = self.files[idx_with_6[idx % len(idx_with_6)] - 154]
        else:
            x = self.files[idx % len(self.files)]
        # print(x)

        if self.data_type =='LR':
            if self.test_scale ==0:
                x = random.choice(x)
            else:
                x = x[self.test_scale-2]

        if self.cache == 'none':
            return x

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

@register('paired-real-image-folders-test')
class PairedRealImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = RealImageFolderTest(root_path_1, **kwargs)
        self.dataset_2 = RealImageFolderTest(root_path_2, **kwargs, data_type='HR')


    def set_test_scale(self,test_scale):
        self.dataset_1.set_test_scale(test_scale)
        self.dataset_2.set_test_scale(test_scale)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        fileh = self.dataset_2[idx]
        filel = self.dataset_1[idx]

        hr_image = Image.open(fileh)
        hr_width, hr_height = hr_image.size
        lr_image = Image.open(filel)
        lr_width, lr_height = lr_image.size
        s = hr_width/lr_width

        w_lr = 48
        x0 = random.randint(0, lr_width - w_lr)
        y0 = random.randint(0, lr_height - w_lr)
        try:
            crop_lr = lr_image.crop([ x0, y0,  x0 + w_lr,y0 + w_lr])
        except: 
            print(filel)
        w_hr = round(w_lr * s)
        x1 = round(x0 * s)
        y1 = round(y0 * s)
        try:
            crop_hr = hr_image.crop([x1,  y1,x1 + w_hr, y1 + w_hr])
        except:
            print(fileh)

        return transforms.ToTensor()(crop_lr.convert('RGB')), transforms.ToTensor()(crop_hr.convert('RGB'))
    
    
    
    
class RealImageFolderRandom(Dataset):

    def __init__(self, root_path,first_k=None,
                 repeat=1, cache='none', data_type='LR'):
        self.repeat = repeat
        self.cache = cache
        self.data_type = data_type


        self.files = []
        filename_list = []
        dirnames = sorted(os.listdir(root_path))

        for dirname in dirnames:
            dir_path = os.path.join(root_path,dirname)
            file_list = sorted([os.path.join(dirname, filename) for filename in os.listdir(dir_path)])
            if file_list:
                filename_list.append(file_list)                  
            
            
        if first_k is not None:
            filename_list = filename_list[:first_k]

        for filenames in filename_list:
            files = []
            # print(filenames)
            for filename in filenames:
                file = os.path.join(root_path, filename)
                files.append(file)

            self.files.append(files)      
                
                            
        

    def __len__(self):
        
        return len(self.files) * self.repeat

    def __getitem__(self, idx):

        x = self.files[idx % len(self.files)]

        x = random.sample(x,2)

        return x

@register('paired-real-image-folders-random')
class PairedRealImageFoldersRandom(Dataset):

    def __init__(self, root_path_1, **kwargs):
        self.dataset_1 = RealImageFolderRandom(root_path_1, **kwargs)


    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        filel,fileh = self.dataset_1[idx]

        lr_image = Image.open(filel)
        hr_image = Image.open(fileh)

        if (hr_image.size[0] < lr_image.size[0]):
            t = hr_image
            hr_image = lr_image
            lr_image = t

        hr_width, hr_height = hr_image.size
        lr_width, lr_height = lr_image.size
        
        s = hr_width/lr_width

        w_lr = 48
        x0 = random.randint(0, lr_width - w_lr)
        y0 = random.randint(0, lr_height - w_lr)
        try:
            crop_lr = lr_image.crop([ x0, y0,  x0 + w_lr,y0 + w_lr])
        except: 
            print(filel)
        w_hr = round(w_lr * s)
        x1 = round(x0 * s)
        y1 = round(y0 * s)
        try:
            crop_hr = hr_image.crop([x1,  y1,x1 + w_hr, y1 + w_hr])
        except:
            print(fileh)

        return transforms.ToTensor()(crop_lr.convert('RGB')), transforms.ToTensor()(crop_hr.convert('RGB'))
    