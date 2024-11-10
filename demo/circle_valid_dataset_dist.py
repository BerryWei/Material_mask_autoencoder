import pandas as pd
import matplotlib.pyplot as plt

# 載入數據
train_file_path = r'D:\2d_composite_mesh_generator\circle_valid\train/revised_descriptors.csv'  # 請更改為你的實際文件路徑
valid_file_path = r'D:\2d_composite_mesh_generator\circle_valid\valid/revised_descriptors.csv'  # 請更改為你的實際文件路徑

train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)

# 設置圖表寬度與分辨率
fig, ax = plt.subplots(figsize=(9.0 / 2.54, 9.0 / 2.54 *0.8), dpi=300)

# 變數名稱與標題
column = 'Vf_real'
title = 'Volume Fraction'

# 繪製直方圖與 PDF
train_data[column].plot(kind='hist', bins=12, density=True, alpha=0.3, color='blue', label='Train Histogram', ax=ax)
train_data[column].plot(kind='density', color='blue', linestyle='-', linewidth=2, label='Train PDF', ax=ax)

valid_data[column].plot(kind='hist', bins=12, density=True, alpha=0.3, color='red', label='Valid Histogram', ax=ax)
valid_data[column].plot(kind='density', color='red', linestyle='--', linewidth=2, label='Valid PDF', ax=ax)

# 設置標籤與標題
ax.set_xlabel(title.capitalize(), fontsize=7, labelpad=2, fontname="Arial")
ax.set_ylabel('Density', fontsize=7, fontname="Arial")
ax.tick_params(axis='both', labelsize=5, direction="in")
ax.legend(fontsize=6, loc='lower right')

# 格式設置
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 調整子圖布局
plt.tight_layout()
plt.savefig(r'D:\Material_mask_autoencoder\results\circle_valid_dist.tiff', format="tiff", dpi=300, bbox_inches='tight')
