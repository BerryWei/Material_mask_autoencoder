import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 定義不同 mask ratio 的路徑
base_dir = "D:/Material_mask_autoencoder/output_dir/"
components = ["C00", "C11", "C22"]
labels = ["$\overline{C}_{1111}$", "$\overline{C}_{2222}$", "$\overline{C}_{1212}$"]

# 保存為 PDF 和 TIFF 文件
output_dir = r"D:\Material_mask_autoencoder\results\pretrained_inclusion\i2i_finetune"
os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

# 定義不同 N_dataset 的數值
N_datasets = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500]

# 設定顏色
colors = cm.tab10.colors[:3]  # 使用 Matplotlib 'tab10' 調色盤中的前三種顏色

# 定義固定 mask ratio
fixed_ratio = "085"

# 儲存每個 N_dataset 的數據
average_r2 = []
component_r2 = {comp: [] for comp in components}

for N_dataset in N_datasets:
    comp_values = []
    for comp in components:
        file_path = os.path.join(base_dir, f"i2i_{N_dataset}_finetune_inclusion_mr{fixed_ratio}_cls_{comp}/log.txt")
        with open(file_path, "r") as f:
            for line in f:
                last_epoch_data = json.loads(line)
            comp_values.append(last_epoch_data["test_r2"])
            component_r2[comp].append(last_epoch_data["test_r2"])
    
    # 計算平均 R2
    average_r2.append(np.mean(comp_values))

# 初始化畫布
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)

# 格式化為千位分隔符號
formatted_ticks = [f'{int(x):,}' for x in N_datasets]

# 第一張子圖：Average R2
axs[0].plot(N_datasets, average_r2, marker='o', markersize=4, label="Average R2", linewidth=1, color='dimgray')
axs[0].set_ylabel("Average R2 score", fontsize=7, fontname="Arial", labelpad=5)
axs[0].set_xlabel("# of data amount", fontsize=7, fontname="Arial")
axs[0].set_xticks(N_datasets)
axs[0].set_ylim([0.85 ,0.975])
axs[0].set_xticklabels(N_datasets)
axs[0].set_xticklabels(formatted_ticks)  # 使用千位分隔格式
axs[0].tick_params(axis='both', labelsize=5, direction="in")
axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
axs[0].set_title(r"Average R2 Score of $\overline{C}_{1111}$, $\overline{C}_{2222}$, $\overline{C}_{1212}$", fontsize=8, fontname="Arial")

# 添加 "a" 標記
axs[0].text(-0.05, 1.05, 'a', transform=axs[0].transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 第二張子圖：C00, C11, C22 的單獨 R2 曲線
for comp, color, label in zip(components, colors, labels):
    axs[1].plot(N_datasets, component_r2[comp], marker='o', markersize=4, label=label, linewidth=1, color=color)

axs[1].set_ylabel("R2 score", fontsize=7, fontname="Arial", labelpad=5)
axs[1].set_xlabel("# of data amount", fontsize=7, fontname="Arial")
axs[1].set_xticks(N_datasets)
axs[1].set_ylim([0.85 ,0.975])

axs[1].set_xticklabels(formatted_ticks)  # 使用千位分隔格式
axs[1].tick_params(axis='both', labelsize=5, direction="in")
axs[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
axs[1].legend(fontsize=6, frameon=False, loc='lower right')
axs[1].set_title(r"R2 Score for $\overline{C}_{1111}$, $\overline{C}_{2222}$, $\overline{C}_{1212}$", fontsize=8, fontname="Arial")

# 添加 "b" 標記
axs[1].text(-0.05, 1.05, 'b', transform=axs[1].transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 緊密布局
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(output_dir, "summary_85percent.pdf"), format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "summary_85percent.tiff"), format="tiff", dpi=300, bbox_inches='tight')
plt.show()
