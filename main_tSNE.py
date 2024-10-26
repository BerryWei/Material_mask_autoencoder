# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import os
import time
from pathlib import Path
import torchvision.datasets as datasets
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torchvision.transforms as transforms
import models_mae

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mmae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--finetune', default=r'D:\Material_mask_autoencoder\output_dir\circle\checkpoint-560.pth', type=str,
                        help='images input size')
    



    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')


    # Dataset parameters
    parser.add_argument('--train_path', default=r'D:\2d_composite_mesh_generator\circle', type=str,
                        help='dataset path')
    parser.add_argument('--valid_path', default=r'D:\2d_composite_mesh_generator\circle_valid', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir/circle_valid',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/circle_valid',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Define the image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (ViT standard input size)
        transforms.ToTensor(),  # Convert PIL.Image to torch.Tensor
    ])


    dataset_train = datasets.ImageFolder(os.path.join(args.train_path), transform=transform)
    dataset_val   = datasets.ImageFolder(os.path.join(args.valid_path), transform=transform)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=20, shuffle=False, num_workers=args.num_workers)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=20, shuffle=False, num_workers=args.num_workers)
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    torch.serialization.add_safe_globals([Namespace])
    checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=True)


    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']


    msg = model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print(msg)

    embeddings = []
    labels = []
    # Process the dataset and calculate embeddings
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in tqdm(data_loader_train, desc="Processing dataset"):
            inputs = inputs.to(device)
            
            # Compute embeddings using model's forward_features method
            embedding_batch = model.forward_features(inputs, mask_ratio=0, global_pool=False)
                # USE CLS token
            # embeddings = model.forward_features(inputs, mask_ratio=0, global_pool=False)

            # # USE global pooling
            # # embeddings = model.forward_features(inputs, mask_ratio=0, global_pool=False)
            
            embeddings.append(embedding_batch.cpu().numpy())  # Move to CPU and store
            labels.append(targets.numpy())

    # Concatenate all embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    np.save(Path(args.output_dir) / 'training_embeddings.npy', embeddings)  # 儲存 embeddings 到 'embeddings.npy'
    np.save(Path(args.output_dir) / 'training_labels.npy', labels)  # 儲存 labels 到 'labels.npy'


#####################
    embeddings = []
    labels = []
    # Process the dataset and calculate embeddings
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in tqdm(data_loader_val, desc="Processing dataset"):
            inputs = inputs.to(device)
            
            # Compute embeddings using model's forward_features method
            embedding_batch = model.forward_features(inputs, mask_ratio=0, global_pool=False)
                # USE CLS token
            # embeddings = model.forward_features(inputs, mask_ratio=0, global_pool=False)

            # # USE global pooling
            # # embeddings = model.forward_features(inputs, mask_ratio=0, global_pool=False)
            
            embeddings.append(embedding_batch.cpu().numpy())  # Move to CPU and store
            labels.append(targets.numpy())

    # Concatenate all embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    np.save(Path(args.output_dir) / 'valid_embeddings.npy', embeddings)  # 儲存 embeddings 到 'embeddings.npy'
    np.save(Path(args.output_dir) / 'valid_labels.npy', labels)  # 儲存 labels 到 'labels.npy'







if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
