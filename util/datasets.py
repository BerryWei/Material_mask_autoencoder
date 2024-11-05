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





def build_transform(is_train, args):

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
    
    return transform



from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import torch

class RegressionImageDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, target_col_name='ERROR', numDataset=None, seed=42):
        """
        img_dir: 影像的路徑
        label_file: 含有影像名稱和對應回歸值的標籤文件
        transform: 影像的轉換操作
        """
        self.img_dir = img_dir
        self.transform = transform
        self.target_col_name = target_col_name
        self.img_labels = pd.read_csv(label_file)
        self.img_labels = self.img_labels.sample(n=numDataset, random_state=seed)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        folder_name = f"mesh{int(self.img_labels.iloc[idx]['Unnamed: 0'])}dir"
        folder_path = os.path.join(self.img_dir, folder_name)
        
        # 加載影像，這裡假設每個資料夾下有一個影像 'image.png'
        img_path = os.path.join(folder_path, 'microstructure.png')
        image = Image.open(img_path).convert("RGB")
        
        # 提取 'Vf_real' 作為回歸標籤
        label = torch.tensor(self.img_labels.iloc[idx][self.target_col_name]/100 , dtype=torch.float16)
        
        if self.transform:
            image = self.transform(image)
        
        
        return image, label