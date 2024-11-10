import os
import json
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.cm as cm

# 定義不同 mask ratio 的路徑
base_dir = "D:/Material_mask_autoencoder/output_dir/"
ratios = "085"
blocks_fine_tuned = [0, 1, 2, 4, 8, 12]
components = ["C00", "C11", "C22"]
# components = ["Vf_real", "C11", "C22"]
labels = ["$\overline{C}_{1111}$", "$\overline{C}_{2222}$", "$\overline{C}_{1212}$"]
# 保存為 PDF 和 TIFF 文件
output_dir = r"D:\Material_mask_autoencoder\results\pretrained_inclusion\i2i_finetune"
os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

# 固定參數
N_dataset = 5000
ratio = "085"  # 固定 Mask Ratio 85%

# 用於存儲數據
data = {comp: [] for comp in components}

# 從文件中讀取數據
for block in blocks_fine_tuned:
    for comp in components:
        if block == 0:
            file_path = os.path.join(base_dir, f"i2i_{N_dataset}_linprobe_inclusion_mr{ratio}_cls_{comp}/log.txt")
        elif block == 12:
            file_path = os.path.join(base_dir, f"i2i_{N_dataset}_finetune_inclusion_mr{ratio}_cls_{comp}/log.txt")
        else:
            file_path = os.path.join(base_dir, f"i2i_{N_dataset}_partialfinetune_{block}_inclusion_mr{ratio}_cls_{comp}/log.txt")
        
        with open(file_path, "r") as f:
            for line in f:
                last_epoch_data = json.loads(line)
            data[comp].append(last_epoch_data["test_r2"])

# 設置圖表大小（雙欄）
fig, axs = plt.subplots(nrows=3, figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)


# 設置圖表寬度，使用單欄（3.54 英寸）或雙欄（7.09 英寸）
# A single column width measures 88 mm and a double column width measures 180 mm
# Figures are best prepared at a width of 90 mm (single column) and 180 mm (double column) with a maximum height of 170mm.

for idx, (comp, ax, label) in enumerate(zip(components, axs, labels)):
    ax.plot(blocks_fine_tuned, data[comp], marker='o', markersize=4, label=label, linewidth=1, color="black")
    ax.set_ylabel(label, fontsize=7, fontname="Arial", labelpad=5)
    ax.tick_params(axis='both', labelsize=7, direction="in")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xticks(blocks_fine_tuned)
    ax.set_xticklabels(blocks_fine_tuned)

    
    # 添加子圖標籤
    ax.text(-0.1, 1.1, chr(97 + idx), transform=ax.transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 全局 X 和 Y 軸標籤
fig.text(0.5, 0.04, "# blocks fine-tuned", ha='center', va='center', fontsize=7, fontname="Arial")  # X軸標籤
fig.text(0.02, 0.5, 'R2 Score', ha='center', va='center', rotation='vertical', fontsize=7, fontname="Arial")  # Y軸標籤

# 緊密布局
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3)

# 保存結果
plt.savefig(os.path.join(output_dir, "r2_vs_blocks_fine_tuned_mr85.pdf"), format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "r2_vs_blocks_fine_tuned_mr85.tiff"), format="tiff", dpi=300, bbox_inches='tight')



##########################
average_r2 = np.mean([data["C00"], data["C11"], data["C22"]], axis=0)

# 設置圖表大小（單欄）
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(18.0 / 2.54, 12.0 / 2.54), dpi=300)  # 寬 180 mm, 高 120 mm
# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9.0 / 2.54, 12.0 / 2.54), dpi=300)  # 寬 90 mm, 高 120 mm


# 第一張子圖：平均 R² 值
axs[0].plot(blocks_fine_tuned, average_r2, marker='o', markersize=4, label="Average R2", linewidth=1, color='dimgray')
axs[0].set_ylabel("Average R² Score", fontsize=7, fontname="Arial", labelpad=5)
axs[0].set_xlabel("# blocks fine-tuned", fontsize=7, fontname="Arial")
axs[0].set_xticks(blocks_fine_tuned)
axs[0].set_xlim([-0.5, 12.5])  # 確保範圍從 0 到 24
axs[0].tick_params(axis='both', labelsize=5, direction="in")
axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# axs[0].set_title(r"Average R² Score of $\overline{C}_{1111}$, $\overline{C}_{2222}$, $\overline{C}_{1212}$", fontsize=8, fontname="Arial")

# 添加 "a" 標記
axs[0].text(-0.1, 1.05, 'a', transform=axs[0].transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 第二張子圖：各 Component R² 曲線
colors = cm.tab10.colors[:3]
for comp, color, label in zip(components, colors, labels):
    axs[1].plot(blocks_fine_tuned, data[comp], marker='o', markersize=4, label=label, linewidth=1, color=color)
axs[1].set_ylabel("R2 score", fontsize=7, fontname="Arial", labelpad=5)
axs[1].set_xlabel("# blocks fine-tuned", fontsize=7, fontname="Arial")
axs[0].set_xticks(blocks_fine_tuned)
axs[1].set_xlim([-0.5, 12.5]) # 確保範圍從 0 到 24
axs[1].tick_params(axis='both', labelsize=5, direction="in")
axs[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].legend(fontsize=6, frameon=False, loc='lower right')
# axs[1].set_title(r"R² Score for $\overline{C}_{1111}$, $\overline{C}_{2222}$, $\overline{C}_{1212}$", fontsize=8, fontname="Arial")

# 添加 "b" 標記
axs[1].text(-0.1, 1.05, 'b', transform=axs[1].transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 緊密布局與保存
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(output_dir, "partial_fine_tuning_SFC.pdf"), format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "partial_fine_tuning_SFC.tiff"), format="tiff", dpi=300, bbox_inches='tight')
plt.show()