import pandas as pd
import matplotlib.pyplot as plt

# 載入數據
train_file_path = r'D:\2d_composite_mesh_generator\inclusion_train_100k\descriptors.csv'  # 請更改為你的實際文件路徑


train_data = pd.read_csv(train_file_path)


# 設置圖表寬度與分辨率
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18.0 / 2.54, 18.0 / 2.54 *0.32), dpi=300)

# 變數名稱列表
columns = ['Vf_real', 'Np', 'Ar']
titles = ['volume fraction', 'number of particles', 'aspect ratio']


# 添加 "a", "b", "c" 標記
labels = ['a', 'b', 'c']
for i, ax in enumerate(axs):
    ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes, fontsize=8, fontweight='bold', ha='center', va='center', fontname="Arial")

# 設定顏色
hist_color = '#a6cee3'  # 淺藍色
pdf_color = '#1f78b4'  # 深藍色

for i, col in enumerate(columns):
    ax = axs[i]
    # 直方圖使用淺藍色
    train_data[col].plot(kind='hist', bins=20, density=True, alpha=0.5, color=hist_color, label='Histogram', ax=ax)
    # 密度曲線使用深藍色
    train_data[col].plot(kind='density', color=pdf_color, linestyle='-', linewidth=2, label='PDF', ax=ax)

    # 設置子圖標籤與標題



    # 設置子圖標籤與標題
    ax.set_xlabel(titles[i].capitalize(), fontsize=7, labelpad=2, fontname="Arial")
    ax.set_ylabel('Density', fontsize=7, fontname="Arial")
    # ax.set_title(titles[i], fontsize=8)
    ax.tick_params(axis='both', labelsize=5, direction="in")
    ax.legend(fontsize=6, loc='lower right', frameon=True)  # 移除圖例邊框

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=5)

# 調整子圖布局
plt.tight_layout()
# plt.show()
plt.savefig(r'D:\Material_mask_autoencoder\results\inclusion_train_dist.tiff', format="tiff", dpi=300, bbox_inches='tight')
