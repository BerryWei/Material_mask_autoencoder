import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 定義不同 mask ratio 的路徑
directories = {
    "mask_ratio_30%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr030_cls_C11/log.txt",
    "mask_ratio_40%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr040_cls_C11/log.txt",
    "mask_ratio_50%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr050_cls_C11/log.txt",
    "mask_ratio_65%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr065_cls_C11/log.txt",
    "mask_ratio_70%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr070_cls_C11/log.txt",
    "mask_ratio_75%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_C11/log.txt",
    "mask_ratio_80%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr080_cls_C11/log.txt",
    "mask_ratio_90%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr090_cls_C11/log.txt",
    "mask_ratio_95%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr095_cls_C11/log.txt",
}

# 創建圖形
plt.figure(figsize=(10, 6))

colormap = cm.viridis
colors = colormap(np.linspace(0, 1, len(directories)))

# 讀取和繪製每個 mask ratio 的 train_loss vs epoch
for (label, file_path), color in zip(directories.items(), colors):
    epochs = []
    train_losses = []
    test_losses = []
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data["epoch"])
            train_losses.append(data["train_loss"] )
            test_losses.append(data["test_loss"] )
    
    plt.plot(epochs, train_losses, label=f"{label} Train", color=color)
    plt.plot(epochs, test_losses, label=f"{label} Valid", color=color, linestyle="--")


# 圖形設定
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Average Loss", fontsize=14)
plt.yscale("log")  # 設置 y 軸為對數刻度
plt.title("", fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()


# 創建圖形
plt.figure(figsize=(10, 6))

colormap = cm.viridis
colors = colormap(np.linspace(0, 1, len(directories)))

# 讀取和繪製每個 mask ratio 的 train_loss vs epoch
for (label, file_path), color in zip(directories.items(), colors):
    epochs = []
    train_r2 = []
    test_r2 = []
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data["epoch"])
            train_r2.append(data["train_r2"] )
            test_r2.append(data["test_r2"])
    
    plt.plot(epochs, train_r2, label=f"{label} Train", color=color)
    plt.plot(epochs, test_r2, label=f"{label} Valid", color=color, linestyle="--")


# 圖形設定
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("R2 score", fontsize=14)
plt.yscale("log")  # 設置 y 軸為對數刻度
plt.title("", fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()


directories = {
    "30%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr030_cls_C11/log.txt",
    "40%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr040_cls_C11/log.txt",
    "50%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr050_cls_C11/log.txt",
    "65%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr065_cls_C11/log.txt",
    "70%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr070_cls_C11/log.txt",
    "75%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_C11/log.txt",
    "80%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr080_cls_C11/log.txt",
    "90%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr090_cls_C11/log.txt",
    "95%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr095_cls_C11/log.txt",
}

# 初始化列表以存儲 mask ratio 和對應的最終 R2 分數
mask_ratios = []
train_r2_final = []
test_r2_final = []

# 讀取每個 mask ratio 的最終 R2 score
for label, file_path in directories.items():
    with open(file_path, "r") as f:
        last_epoch_data = None
        for line in f:
            last_epoch_data = json.loads(line)  # 取得最後一行的數據
        
        # 獲取最後一個 epoch 的 train_r2 和 test_r2
        if last_epoch_data:
            mask_ratios.append(int(label.strip('%')))
            train_r2_final.append(last_epoch_data["train_r2"])
            test_r2_final.append(last_epoch_data["test_r2"])

# 繪製圖形
plt.figure(figsize=(10, 6))
plt.plot(mask_ratios, train_r2_final, marker='o', label="Train R2")
plt.plot(mask_ratios, test_r2_final, marker='o', linestyle="--", label="Valid R2")

# 圖形設定
plt.xlabel("Mask Ratio (%)", fontsize=14)
plt.ylabel("Final R2 Score", fontsize=14)
plt.title("Final R2 Score vs Mask Ratio", fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()