import os
import json
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.cm as cm

# 定義不同 mask ratio 的路徑
base_dir = "D:/Material_mask_autoencoder/output_dir/"
ratios = ["010", "015", "020", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070", "075", "080", "085", "090", "095"]
components = [ "C00", "C11", "C22"]
# components = ["Vf_real", "C11", "C22"]
labels = [ "$\overline{C}_{1111}$", "$\overline{C}_{2222}$", "$\overline{C}_{1212}$"]
# 保存為 PDF 和 TIFF 文件
output_dir = r"D:\Material_mask_autoencoder\results\pretrained_circle\c2c_linear_probing"
os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

N_dataset = 5000

# 用於存儲數據
data = {comp: [] for comp in components}
mask_ratios = []

# 從文件中讀取數據
for ratio in ratios:
    mask_ratios.append(int(ratio))
    for comp in components:
        file_path = os.path.join(base_dir, f"c2c_{N_dataset}_linprobe_circular_mr{ratio}_cls_{comp}/log.txt")
        with open(file_path, "r") as f:
            for line in f:
                last_epoch_data = json.loads(line)
            data[comp].append(last_epoch_data["test_r2"])

# 設置圖表寬度，使用單欄（3.54 英寸）或雙欄（7.09 英寸）
# A single column width measures 88 mm and a double column width measures 180 mm
# Figures are best prepared at a width of 90 mm (single column) and 180 mm (double column) with a maximum height of 170mm.

fig, axs = plt.subplots(nrows=3, figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)

# 子圖的設置
for idx, (comp, ax, label) in enumerate(zip(components, axs, labels)):
    ax.plot(mask_ratios, data[comp], marker='o', markersize=4, label=label, linewidth=1, color="black")
    ax.set_ylabel(label, fontsize=7, fontname="Arial", labelpad=5)
    # ax.legend(fontsize=8, loc='lower right', frameon=False)
    
    ax.tick_params(axis='both', labelsize=7, direction="in")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# X軸和標題設置

# 使用 fig.text() 手動設置全局 X 和 Y 軸標籤
fig.text(0.5, 0.04, "Mask Ratio (%)", ha='center', va='center', fontsize=8, fontname="Arial")  # X軸標籤
fig.text(0.02, 0.5, 'R2 Score', ha='center', va='center', rotation='vertical', fontsize=8, fontname="Arial")  # Y軸標籤


# fig.suptitle("Final R2 Score vs Mask Ratio for C00, C11, and C22", fontsize=9, fontname="Arial", y=1.02)

# 進行緊密布局
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3) 

plt.savefig(os.path.join(output_dir, "summary_0.pdf"), format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "summary_0.tiff"), format="tiff", dpi=300, bbox_inches='tight')
plt.show()


# 計算 C00, C11, C22 的平均 R2 值
average_r2 = np.mean([data["C00"], data["C11"], data["C22"]], axis=0)

# 設置圖表寬度，使用單欄（3.54 英寸）或雙欄（7.09 英寸）
# A single column width measures 88 mm and a double column width measures 180 mm
# Figures are best prepared at a width of 90 mm (single column) and 180 mm (double column) with a maximum height of 170mm.

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)

# fig, axs = plt.subplots(nrows=3, figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)

# 定義顏色
colors = cm.tab10.colors[:3]  # 使用 Matplotlib 'tab10' 調色盤中的前三種顏色

# 第一張子圖：平均 R2 值
axs[0].plot(mask_ratios, average_r2, marker='o', markersize=4, label="Average R2", linewidth=1, color='dimgray')
axs[0].set_ylabel("Average R2 Score", fontsize=7, fontname="Arial", labelpad=5)
axs[0].set_xlabel("Mask Ratio (%)", fontsize=7, fontname="Arial")
axs[0].tick_params(axis='both', labelsize=5, direction="in")
axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
axs[0].set_title(r"Average R2 Score of $\overline{C}_{1111}$, $\overline{C}_{2222}$, $\overline{C}_{1212}$", fontsize=8, fontname="Arial")

# 添加 "a" 標記
axs[0].text(-0.05, 1.05, 'a', transform=axs[0].transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")


# 第二張子圖：C00, C11, C22 的單獨 R2 曲線
for comp, color in zip(components, colors):
    axs[1].plot(mask_ratios, data[comp], marker='o', markersize=4, label=comp, linewidth=1, color=color)
axs[1].set_ylabel("R2 Score", fontsize=7, fontname="Arial", labelpad=5)
axs[1].set_xlabel("Mask Ratio (%)", fontsize=7, fontname="Arial")
axs[1].tick_params(axis='both', labelsize=5, direction="in")
axs[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].legend(fontsize=6, frameon=False, loc='lower right')
axs[1].set_title(r"R2 Score for $\overline{C}_{1111}$, $\overline{C}_{2222}$, $\overline{C}_{1212}$", fontsize=8, fontname="Arial")


# 添加 "b" 標記
axs[1].text(-0.05, 1.05, 'b', transform=axs[1].transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 緊密布局
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(output_dir, "summary_1.pdf"), format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "summary_1.tiff"), format="tiff", dpi=300, bbox_inches='tight')
plt.show()