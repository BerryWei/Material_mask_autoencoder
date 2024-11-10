import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 定義不同 mask ratio 的路徑
base_dir = "D:/Material_mask_autoencoder/output_dir/"
ratios = ["010", "015", "020", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070", "075", "080", "085", "090", "095"]
components = ["C00", "C11", "C22"]
labels = ["$\overline{C}_{1111}$", "$\overline{C}_{2222}$", "$\overline{C}_{1212}$"]

# 保存為 PDF 和 TIFF 文件
output_dir = r"D:\Material_mask_autoencoder\results\pretrained_inclusion\i2i_finetune"
os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

# 定義不同 N_dataset 的數值
N_datasets = [5000, 625, 312]

# 設定 viridis colormap
colormap = cm.viridis(np.linspace(0, 1, len(N_datasets)))

# 初始化畫布，將 nrows 改為 4
fig, axs = plt.subplots(nrows=4, figsize=(18.0 / 2.54, 18.0 / 2.54 * 0.7), dpi=300)

# 初始化平均值子圖的數據
average_data = {N_dataset: [] for N_dataset in N_datasets}

# 設置標記 a, b, c, d
subplot_labels = ['a', 'b', 'c', 'd']

# 計算每個 mask ratio 的平均值，並繪製子圖
for N_dataset, color in zip(N_datasets, colormap):
    mask_ratios = []
    avg_values = []

    for ratio in ratios:
        mask_ratios.append(int(ratio))
        comp_values = []

        for comp in components:
            file_path = os.path.join(base_dir, f"i2i_{N_dataset}_finetune_inclusion_mr{ratio}_cls_{comp}/log.txt")
            with open(file_path, "r") as f:
                for line in f:
                    last_epoch_data = json.loads(line)
                comp_values.append(last_epoch_data["test_r2"])

        avg_values.append(np.mean(comp_values))  # 計算平均值
    average_data[N_dataset] = avg_values

    axs[0].plot(mask_ratios, avg_values, marker='o', markersize=4, label=f"N={N_dataset}", linewidth=1, color=color)

axs[0].set_ylabel("Average R2", fontsize=7, fontname="Arial", labelpad=5)
axs[0].tick_params(axis='both', labelsize=7, direction="in")
axs[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# axs[0].legend(fontsize=6, loc='lower right', frameon=True)

# 在子圖左上角添加 a, b, c, d
for i, ax in enumerate(axs):
    ax.text(-0.065, 1.1, subplot_labels[i], transform=ax.transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 子圖的設置
for idx, (comp, ax, label) in enumerate(zip(components, axs[1:], labels)):
    for N_dataset, color in zip(N_datasets, colormap):
        data = []
        mask_ratios = []

        # 從文件中讀取數據
        for ratio in ratios:
            mask_ratios.append(int(ratio))
            file_path = os.path.join(base_dir, f"i2i_{N_dataset}_finetune_inclusion_mr{ratio}_cls_{comp}/log.txt")
            with open(file_path, "r") as f:
                for line in f:
                    last_epoch_data = json.loads(line)
                data.append(last_epoch_data["test_r2"])

        ax.plot(mask_ratios, data, marker='o', markersize=4, label=f"N={N_dataset}", linewidth=1, color=color)

    ax.set_ylabel(label, fontsize=7, fontname="Arial", labelpad=5)
    ax.tick_params(axis='both', labelsize=7, direction="in")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    #ax.legend(fontsize=6, loc='lower right', frameon=True)  # 添加圖例

# 使用 fig.text() 手動設置全局 X 和 Y 軸標籤
fig.text(0.5, 0.04, "Mask Ratio (%)", ha='center', va='center', fontsize=8, fontname="Arial")  # X軸標籤
fig.text(0.02, 0.5, 'R2 Score', ha='center', va='center', rotation='vertical', fontsize=8, fontname="Arial")  # Y軸標籤


# 將圖例從子圖中移出，放置在整個圖表右側
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6, frameon=False, bbox_to_anchor=(0.99, 0.5), 
           title="dataset size", title_fontsize=7, borderaxespad=0.5, labelspacing=1.3)



# 緊密布局
plt.tight_layout(pad=2.0)
plt.subplots_adjust(hspace=0.3)


# 保存結果
plt.savefig(os.path.join(output_dir, "varies_dataset_i2i.pdf"), format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "varies_dataset_i2i.tiff"), format="tiff", dpi=300, bbox_inches='tight')

plt.show()