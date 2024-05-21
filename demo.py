import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test_real import batched_predict
from datasets.image_folder import RealImageFolderTest

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--input_path')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output_path')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from datasets.image_folder import RealImageFolderTest
    
    # img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    
    LR_ds = RealImageFolderTest(os.path.join(args.input_path, 'test_LR/'))
    # HR_ds = RealImageFolderTest(os.path.join(args.input_path, 'test_HR/'), data_type='HR')
    
    LR_ds.set_test_scale(int(args.scale))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    

    for i in tqdm(range(len(LR_ds))):
        image_path = LR_ds[i]
        img = transforms.ToTensor()(Image.open(image_path).convert('RGB'))
        h = int(img.shape[-2] * int(args.scale))
        w = int(img.shape[-1] * int(args.scale))
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        transforms.ToPILImage()(pred).save(os.path.join(args.output_path, f"{image_path.split('/')[-2]}_{args.name}.JPG"))
