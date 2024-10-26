# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import torch

class RegressionImageDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        """
        img_dir: 影像的路徑
        label_file: 含有影像名稱和對應回歸值的標籤文件
        transform: 影像的轉換操作
        """
        self.img_dir = img_dir
        self.transform = transform

        self.img_labels = pd.read_csv(label_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        folder_name = f"mesh{int(self.img_labels.iloc[idx]['Unnamed: 0'])}dir"
        folder_path = os.path.join(self.img_dir, folder_name)
        
        # 加載影像，這裡假設每個資料夾下有一個影像 'image.png'
        img_path = os.path.join(folder_path, 'microstructure.png')
        image = Image.open(img_path).convert("RGB")
        
        # 提取 'Vf_real' 作為回歸標籤
        label = torch.tensor(self.img_labels.iloc[idx]['Vf_real'] / 100, dtype=torch.float16)
        
        if self.transform:
            image = self.transform(image)
        
        
        return image, label