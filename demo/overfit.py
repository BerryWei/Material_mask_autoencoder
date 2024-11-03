import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 定義不同 mask ratio 的路徑
directories = {
    "w/ BN;w/ DA": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real/log.txt",
    "w/o BN;w/o DA": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real_NoBN_NoDataAug/log.txt",
    "w/o BN;w/ DA": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real_NoBN_wDataAug/log.txt",
    "w/ BN;w/o DA": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real_wBN_woDataAug/log.txt",
    "w/ BN;w/ DA(inverse dataset)": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real_wBN_wDataAug_invDataset/log.txt",
    "w/o BN;w/o DA(inverse dataset)": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real_woBN_woDataAug_invDataset/log.txt",
    "w/o BN;w/ DA(inverse dataset)": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real_woBN_wDataAug_invDataset/log.txt",
    "w/ BN;w/o DA(inverse dataset)": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_Vf_real_wBN_woDataAug_invDataset/log.txt",
    
    
    
    
}

# 創建圖形
plt.figure(figsize=(10, 6))

colormap = cm.tab10
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
plt.legend(fontsize=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()