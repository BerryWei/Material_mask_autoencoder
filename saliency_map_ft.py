import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision.transforms import ToPILImage
import pandas as pd
from models_vit import vit_base_patch16_material

# 模型載入函數
def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    model = vit_base_patch16_material(num_classes=1, drop_path_rate=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

# 計算 saliency map 函數
def compute_saliency_map(model: torch.nn.Module, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    input_tensor.requires_grad_()
    if input_tensor.shape[1] == 3:  # 若為 RGB，轉換為灰階
        input_tensor = input_tensor.mean(dim=1, keepdim=True)
    
    output = model(input_tensor)
    criterion = torch.nn.MSELoss()
    mse_loss = criterion(output, target.float())
    gradients = torch.autograd.grad(outputs=mse_loss, inputs=input_tensor, create_graph=True)[0]
    saliency_map = gradients.abs().squeeze().detach().cpu()
    return saliency_map

# 繪製單行 Saliency Map 圖片
def plot_saliency_row(axs, original_image, saliency_map, cmap='viridis', vmax=0.1, gamma=1.0):
    """
    繪製原圖、Saliency Map 與疊加圖。
    """
    to_pil = ToPILImage()
    original_gray = to_pil(original_image.squeeze(0))  # 灰階圖

    # Gamma 校正
    saliency_map = torch.clamp(saliency_map, min=0)
    saliency_map_normalized = torch.pow(saliency_map, gamma) / torch.pow(saliency_map.max(), gamma)

    # Saliency Map
    axs[0].imshow(saliency_map_normalized.numpy(), cmap=cmap, vmin=0, vmax=vmax)
    axs[0].axis('off')
    # axs[0].set_title("Saliency Map", fontsize=7)

    # 疊加圖
    overlay = np.array(original_gray.convert("RGBA"))
    saliency_colormap = plt.cm.get_cmap(cmap)(np.clip(saliency_map_normalized.numpy() / vmax, 0, 1))[:, :, :3]
    saliency_overlay = (saliency_colormap * 255).astype(np.uint8)
    saliency_overlay = Image.fromarray(saliency_overlay).convert('RGBA')
    saliency_overlay_np = np.array(saliency_overlay)
    saliency_overlay_np[:, :, 3] = 128  # 設定透明度
    blended = Image.alpha_composite(Image.fromarray(overlay), Image.fromarray(saliency_overlay_np))

    axs[1].imshow(blended)
    axs[1].axis('off')
    # axs[1].set_title("Overlay", fontsize=7)

# 圖像轉換
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# 路徑與參數
image_paths = [
    r'D:\2d_composite_mesh_generator\inclusion_valid\valid\mesh0dir\microstructure.png',
    r'D:\2d_composite_mesh_generator\inclusion_valid\valid\mesh1dir\microstructure.png',
    r'D:\2d_composite_mesh_generator\inclusion_valid\valid\mesh2dir\microstructure.png',
    r'D:\2d_composite_mesh_generator\inclusion_valid\valid\mesh3dir\microstructure.png'
]

target_labels = ['C00', 'C11', 'C22']
checkpoints = [
    r'D:\Material_mask_autoencoder\output_dir\i2i_5000_finetune_inclusion_mr085_cls_C00\checkpoint-599.pth',
    r'D:\Material_mask_autoencoder\output_dir\i2i_5000_finetune_inclusion_mr085_cls_C11\checkpoint-599.pth',
    r'D:\Material_mask_autoencoder\output_dir\i2i_5000_finetune_inclusion_mr085_cls_C22\checkpoint-599.pth'
]
csv_path = r'D:\2d_composite_mesh_generator\inclusion_valid\valid\revised_descriptors.csv'
device = 'cuda'

# 繪圖設置
fig = plt.figure(figsize=(18 / 2.54, 10 / 2.54), dpi=300)
gs = GridSpec(len(image_paths), 7, figure=fig, wspace=0.2, hspace=0.4)

# 加載標籤
img_labels = pd.read_csv(csv_path)

# 繪製每個樣本的多目標
for i, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    # ax.set_title(f"Original #{i+1}", fontsize=7)

    # 在最左側加上 Instance 標籤
    bbox0 = ax.get_position(fig)
    y_center = (bbox0.y0 + bbox0.y1 ) / 2  # 計算 y 軸中心位置


    
    


    fig.text(0.1, y_center, f"Instance #{i + 1}",  # 調整 x 位置
             fontsize=8, fontweight='bold', fontname="Arial", va='center', ha='right', rotation='vertical',
             transform=fig.transFigure)




    for j, (target_label, checkpoint_path) in enumerate(zip(target_labels, checkpoints)):
        model = load_model(checkpoint_path, device)
        labels = img_labels[target_label][:len(image_paths)] / 100  # 標準化
        target = torch.tensor([[labels[i]]], device=device, dtype=torch.float32)
        saliency_map = compute_saliency_map(model, input_tensor, target)

        axs = [fig.add_subplot(gs[i, j * 2 + k + 1]) for k in range(2)]
        plot_saliency_row(axs, input_tensor.squeeze(0).cpu(), saliency_map, vmax=0.3, cmap='jet')


# fig.subplots_adjust(top=0.85)  # 預留空間，值越小留白越多

# 定義每個標題的位置與內容
titles = [
    ("Original", ""),
    ("Saliency Map", r"$(\overline{C}_{1111})$"),
    ("Overlay", r"$(\overline{C}_{1111})$"),
    ("Saliency Map", r"$(\overline{C}_{2222})$"),
    ("Overlay", r"$(\overline{C}_{2222})$"),
    ("Saliency Map", r"$(\overline{C}_{1212})$"),
    ("Overlay", r"$(\overline{C}_{1212})$")
]

# 定義每個標題的 x 位置 (以列為基準)
title_positions = np.linspace(0.175, 0.85, 7)  # 均分7個位置，略微調整邊界

# # 在圖上方依序添加每個標題
# for pos, title in zip(title_positions, titles):
#     fig.text(pos, 0.92, title,  # y設為0.98，調整標題位置
#              ha='center', va='top', fontsize=7, fontweight='bold', fontname="Arial", linespacing=1.5)
    
for pos, (title_top, title_bottom) in zip(title_positions, titles):
    if title_top == "Original":
        fig.text(pos, 0.92, title_top,  # 上标题位置
             ha='center', va='bottom', fontsize=8, fontweight='bold', fontname="Arial")
    else:
        fig.text(pos, 0.93, title_top,  # 上标题位置
                ha='center', va='bottom', fontsize=7, fontweight='bold', fontname="Arial")
        
        fig.text(pos, 0.92, title_bottom,  # 下标题位置
             ha='center', va='top', fontsize=7, fontname="Arial", style='italic')


# fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig(r'D:\Material_mask_autoencoder\results\saliency_visualization_7cols.tiff', format="tiff", dpi=300, bbox_inches='tight')
plt.savefig(r'D:\Material_mask_autoencoder\results\saliency_visualization_7cols.pdf', format="pdf", dpi=300, bbox_inches='tight')

# plt.show()
