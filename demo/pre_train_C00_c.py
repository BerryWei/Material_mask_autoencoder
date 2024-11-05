import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 定義不同 mask ratio 的路徑
directories = {
    "10%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr010_cls_C00/log.txt",
    "15%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr015_cls_C00/log.txt",
    "20%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr020_cls_C00/log.txt",
    "25%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr025_cls_C00/log.txt",
    "30%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr030_cls_C00/log.txt",
    "35%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr035_cls_C00/log.txt",
    "40%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr040_cls_C00/log.txt",
    "45%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr045_cls_C00/log.txt",
    "50%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr050_cls_C00/log.txt",
    "55%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr055_cls_C00/log.txt",
    "60%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr060_cls_C00/log.txt",
    "65%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr065_cls_C00/log.txt",
    "70%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr070_cls_C00/log.txt",
    "75%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr075_cls_C00/log.txt",
    "80%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr080_cls_C00/log.txt",
    "85%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr085_cls_C00/log.txt",
    "90%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr090_cls_C00/log.txt",
    "95%": "D:/Material_mask_autoencoder/output_dir/linprobe_circular_mr095_cls_C00/log.txt",
}

# 創建訓練和驗證曲線圖的函數
def plot_loss_vs_epoch(directories, ylabel, value_key, log_scale=False):
    plt.figure(figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)  # 雙欄寬度，保持適合的比例
    colormap = cm.viridis
    colors = colormap(np.linspace(0, 1, len(directories)))
    for (label, file_path), color in zip(directories.items(), colors):
        epochs, train_vals, test_vals = [], [], []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                epochs.append(data["epoch"])
                train_vals.append(data[f"train_{value_key}"])
                test_vals.append(data[f"test_{value_key}"])
        plt.plot(epochs, train_vals, label=f"{label} Train", color=color)
        plt.plot(epochs, test_vals, label=f"{label} Valid", color=color, linestyle="--")
    plt.xlabel("Epoch", fontsize=7)
    plt.ylabel(ylabel, fontsize=7)
    if log_scale:
        plt.yscale("log")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.show()

# 繪製平均損失和 R2 score 的圖
plot_loss_vs_epoch(directories, "Average Loss", "loss", log_scale=True)
plot_loss_vs_epoch(directories, "R2 Score", "r2", log_scale=True)

# 最終 R2 分數 vs Mask Ratio 圖
mask_ratios, train_r2_final, test_r2_final = [], [], []
for label, file_path in directories.items():
    with open(file_path, "r") as f:
        last_epoch_data = None
        for line in f:
            last_epoch_data = json.loads(line)
        if last_epoch_data:
            mask_ratios.append(int(label.strip('%')))
            train_r2_final.append(last_epoch_data["train_r2"])
            test_r2_final.append(last_epoch_data["test_r2"])



plt.figure(figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)  # 雙欄寬度，保持適合的比例
plt.plot(mask_ratios, train_r2_final, marker='o', markersize=4, label="Train R2", linewidth=1)
plt.plot(mask_ratios, test_r2_final, marker='o', markersize=4, linestyle="--", label="Valid R2", linewidth=1)


plt.xlabel("Mask Ratio (%)", fontsize=7, fontname="Arial")
plt.ylabel("Final R2 Score", fontsize=7, fontname="Arial")
plt.title("Final R2 Score vs Mask Ratio", fontsize=7, fontname="Arial", pad=10)
plt.legend(fontsize=7, loc='lower right', frameon=False)
plt.xticks(fontsize=6, fontname="Arial")
plt.yticks(fontsize=6, fontname="Arial")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()