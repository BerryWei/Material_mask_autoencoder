import sys
import os
import requests
from pathlib import Path
import torch
import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image

import models_mae

def run_one_image_with_axes(img, model, ax, titles):
    x = torch.tensor(img)  # [h, w, c]
    x = x.unsqueeze(dim=0)  # [n, h, w, c]
    x = torch.einsum('nhwc->nchw', x)  # [n, c, h, w]
    if x.shape[1] == 3:
        x = x.mean(dim=1, keepdim=True)  # RGB 轉灰階
    loss, y, mask = model(x.float(), mask_ratio=0.85)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    mask = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 1)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', x)
    im_masked = x * (1 - mask)
    green_mask = torch.zeros(1, 224, 224, 3)
    green_mask[:, :, :, :] = torch.tensor([0.5, 0.7, 1.0]).view(1, 1, 1, 3)
    im_masked_colored = im_masked + green_mask * mask
    im_paste = x * (1 - mask) + y * mask

    # 展示圖片於特定的 ax
    show_image(x[0].repeat(1, 1, 3), titles[0], ax=ax[0])
    show_image(im_masked_colored[0], titles[1], ax=ax[1])
    show_image(im_paste[0].repeat(1, 1, 3), titles[2], ax=ax[2])


def show_image(image, title='', ax=None):
    # Display image with matplotlib axes
    assert image.shape[2] == 3
    if ax is None:
        plt.imshow(torch.clip(image * 255, 0, 255).int())
        plt.title(title, fontsize=7, fontname="Arial")
        plt.axis('off')
    else:
        ax.imshow(torch.clip(image * 255, 0, 255).int())
        ax.set_title(title, fontsize=8, fontname="Arial", pad=5)  # Add padding to title
        ax.axis('off')


def prepare_model(chkpt_dir, arch='mmae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

chkpt_dir = Path(r'D:\Material_mask_autoencoder\output_dir\inclusion_mr085\checkpoint-399.pth')
model_mae = prepare_model(chkpt_dir, arch='mmae_vit_base_patch16')
print('Model loaded.')

# 定義兩欄大圖的設置




transform_train = transforms.Compose([
            transforms.ToTensor()])
dataset_train = datasets.ImageFolder(os.path.join(r'D:\2d_composite_mesh_generator\circle_valid\train'), transform=transform_train)
dataset_valid = datasets.ImageFolder(os.path.join(r'D:\2d_composite_mesh_generator\circle_valid\valid'), transform=transform_train)

# 設置圖表寬度，使用單欄（3.54 英寸）或雙欄（7.09 英寸）
# A single column width measures 88 mm and a double column width measures 180 mm
# Figures are best prepared at a width of 90 mm (single column) and 180 mm (double column) with a maximum height of 170mm.
from matplotlib.gridspec import GridSpec
num_instances = 4

# Define figure and GridSpec
fig = plt.figure(figsize=(9. / 2.54, 9. / 2.54 *1.8), dpi=300)
gs = fig.add_gridspec(num_instances, 3)  # Use add_gridspec for finer control

axes = []
for i in range(num_instances):
    row_axes = [
        fig.add_subplot(gs[i, 0]),  # Original
        fig.add_subplot(gs[i, 1]),  # Masked
        fig.add_subplot(gs[i, 2])   # Reconstruction
    ]
    axes.append(row_axes)
fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.3)
for i, row_axes in enumerate(axes):
    # Adjust Instance label position
    bbox0 = row_axes[0].get_position(fig)
    bbox1 = row_axes[2].get_position(fig)
    y_center = (bbox0.y0 + bbox1.y1) / 2 

    # Move "Instance" label slightly to the right
    fig.text(0.09, y_center, f"Instance #{i + 1}",  # Adjusted x for better positioning
             fontsize=8, fontweight='bold', fontname="Arial", va='center', ha='right', rotation='vertical',
             transform=fig.transFigure)

    # Process training images
    img_tensor, _ = dataset_train[i]
    img_tensor = torch.einsum('chw->hwc', img_tensor)  # Adjust tensor dimensions
    torch.manual_seed(2)
    run_one_image_with_axes(img_tensor, model_mae, row_axes[0:3],
                            ["Original", "Masked", "Reconstruction"])

# Manually adjust subplot spacing
# fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.01, hspace=0.3)

# Save results
# plt.tight_layout()
plt.savefig(r'D:\Material_mask_autoencoder\results\pretrained_inclusion\reconstruction_circle.png')
plt.show()