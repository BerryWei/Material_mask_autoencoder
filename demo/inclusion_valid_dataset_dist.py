import pandas as pd
import matplotlib.pyplot as plt

# 載入數據
train_file_path = r'D:\2d_composite_mesh_generator\inclusion_valid\train/revised_descriptors.csv'  # 請更改為你的實際文件路徑
valid_file_path = r'D:\2d_composite_mesh_generator\inclusion_valid\valid/revised_descriptors.csv'  # 請更改為你的實際文件路徑

train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)

# 設置圖表寬度與分辨率
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18.0 / 2.54, 18.0 / 2.54 *0.32), dpi=300)

# 變數名稱列表
columns = ['Vf_real', 'Np', 'Ar']
titles = ['volume fraction', 'number of particles', 'aspect ratio']


# 添加 "a", "b", "c" 標記
labels = ['a', 'b', 'c']
for i, ax in enumerate(axs):
    ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")


for i, col in enumerate(columns):
    ax = axs[i]
    train_data[col].plot(kind='hist', bins=12, density=True, alpha=0.3, color='blue', label='Train Histogram', ax=ax)
    train_data[col].plot(kind='density', color='blue', linestyle='-', linewidth=2, label='Train PDF', ax=ax)

    valid_data[col].plot(kind='hist', bins=12, density=True, alpha=0.3, color='red', label='Valid Histogram', ax=ax)
    valid_data[col].plot(kind='density', color='red', linestyle='--', linewidth=2, label='Valid PDF', ax=ax)
    


    # 設置子圖標籤與標題
    ax.set_xlabel(titles[i].capitalize(), fontsize=7, labelpad=2, fontname="Arial")
    ax.set_ylabel('Density', fontsize=7, fontname="Arial")
    # ax.set_title(titles[i], fontsize=8)
    ax.tick_params(axis='both', labelsize=5, direction="in")
    ax.legend(fontsize=6, loc='lower right')

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=5)

# 調整子圖布局
plt.tight_layout()
plt.savefig(r'D:\Material_mask_autoencoder\results\inclusion_valid_dist.tiff', format="tiff", dpi=300, bbox_inches='tight')
