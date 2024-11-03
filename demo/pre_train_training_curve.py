import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 定義不同 mask ratio 的路徑
directories = {
    "mask_ratio_20%": "D:/Material_mask_autoencoder/output_dir/circle_mr020/log.txt",
    "mask_ratio_30%": "D:/Material_mask_autoencoder/output_dir/circle_mr030/log.txt",
    "mask_ratio_40%": "D:/Material_mask_autoencoder/output_dir/circle_mr040/log.txt",
    "mask_ratio_50%": "D:/Material_mask_autoencoder/output_dir/circle_mr050/log.txt",
    "mask_ratio_60%": "D:/Material_mask_autoencoder/output_dir/circle_mr060/log.txt",
    "mask_ratio_65%": "D:/Material_mask_autoencoder/output_dir/circle_mr065/log.txt",
    "mask_ratio_70%": "D:/Material_mask_autoencoder/output_dir/circle_mr070/log.txt",
    "mask_ratio_75%": "D:/Material_mask_autoencoder/output_dir/circle_mr075/log.txt",
    "mask_ratio_80%": "D:/Material_mask_autoencoder/output_dir/circle_mr080/log.txt",
    "mask_ratio_90%": "D:/Material_mask_autoencoder/output_dir/circle_mr090/log.txt",
}

# 創建圖形
plt.figure(figsize=(10, 6))

# 讀取和繪製每個 mask ratio 的 train_loss vs epoch
# for label, file_path in directories.items():
#     epochs = []
#     train_losses = []
    
#     with open(file_path, "r") as f:
#         for line in f:
#             data = json.loads(line)
#             epochs.append(data["epoch"])
#             train_losses.append(data["train_loss"])
    
#     plt.plot(epochs, train_losses, label=label)



colormap = cm.viridis
colors = colormap(np.linspace(0, 1, len(directories)))

# 讀取和繪製每個 mask ratio 的 train_loss vs epoch
for (label, file_path), color in zip(directories.items(), colors):
    epochs = []
    train_losses = []
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data["epoch"])
            train_losses.append(data["train_loss"])
    
    plt.plot(epochs, train_losses, label=label, color=color)


# 圖形設定
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Train Loss", fontsize=14)
plt.yscale("log")  # 設置 y 軸為對數刻度
plt.title("Training Curves(pre-trained on circle inclusion)", fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()
