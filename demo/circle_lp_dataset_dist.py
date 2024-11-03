import pandas as pd
import matplotlib.pyplot as plt

# 載入數據
train_file_path = r'D:\2d_composite_mesh_generator\circle_valid_test\train/revised_descriptors.csv'  # 請更改為你的實際文件路徑
valid_file_path = r'D:\2d_composite_mesh_generator\circle_valid_test\valid/revised_descriptors.csv'  # 請更改為你的實際文件路徑

train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)

# 繪製柱狀圖和 PDF 疊加圖
plt.figure(figsize=(10, 6))

# 繪製 train 數據的直方圖和 PDF
train_data['Vf_real'].plot(kind='hist', bins=12, density=True, alpha=0.3, color='blue', label='Train Histogram')
train_data['Vf_real'].plot(kind='density', color='blue', linestyle='-', linewidth=2, label='Train PDF')

# 繪製 valid 數據的直方圖和 PDF
valid_data['Vf_real'].plot(kind='hist', bins=12, density=True, alpha=0.3, color='red', label='Valid Histogram')
valid_data['Vf_real'].plot(kind='density', color='red', linestyle='--', linewidth=2, label='Valid PDF')

# 設置圖表標籤和標題
plt.xlabel('Vf_real (Volume Fraction)', fontsize=14)
plt.ylabel('Density', fontsize=14)  # 保持為概率密度
plt.title('Histogram and PDF of Vf_real (Volume Fraction) for Train and Valid', fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()

# Display the range of 'Vf_real' for both train and valid datasets

train_vf_real_min, train_vf_real_max = train_data['Vf_real'].min(), train_data['Vf_real'].max()
valid_vf_real_min, valid_vf_real_max = valid_data['Vf_real'].min(), valid_data['Vf_real'].max()

print(train_vf_real_min, train_vf_real_max, valid_vf_real_min, valid_vf_real_max)