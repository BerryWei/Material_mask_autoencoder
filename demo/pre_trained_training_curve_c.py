import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 定義不同 mask ratio 的路徑
directories = {
    "10%": "D:/Material_mask_autoencoder/output_dir/circle_mr010/log.txt",
    "15%": "D:/Material_mask_autoencoder/output_dir/circle_mr015/log.txt",
    "20%": "D:/Material_mask_autoencoder/output_dir/circle_mr020/log.txt",
    "25%": "D:/Material_mask_autoencoder/output_dir/circle_mr025/log.txt",
    "30%": "D:/Material_mask_autoencoder/output_dir/circle_mr030/log.txt",
    "35%": "D:/Material_mask_autoencoder/output_dir/circle_mr035/log.txt",
    "40%": "D:/Material_mask_autoencoder/output_dir/circle_mr040/log.txt",
    "50%": "D:/Material_mask_autoencoder/output_dir/circle_mr050/log.txt",
    "55%": "D:/Material_mask_autoencoder/output_dir/circle_mr055/log.txt",
    "60%": "D:/Material_mask_autoencoder/output_dir/circle_mr060/log.txt",
    "65%": "D:/Material_mask_autoencoder/output_dir/circle_mr065/log.txt",
    "70%": "D:/Material_mask_autoencoder/output_dir/circle_mr070/log.txt",
    "75%": "D:/Material_mask_autoencoder/output_dir/circle_mr075/log.txt",
    "80%": "D:/Material_mask_autoencoder/output_dir/circle_mr080/log.txt",
    "85%": "D:/Material_mask_autoencoder/output_dir/circle_mr085/log.txt",
    "90%": "D:/Material_mask_autoencoder/output_dir/circle_mr090/log.txt",
    "95%": "D:/Material_mask_autoencoder/output_dir/circle_mr095/log.txt",
}

# 設置圖表寬度，使用單欄（3.54 英寸）或雙欄（7.09 英寸）
# A single column width measures 88 mm and a double column width measures 180 mm
# Figures are best prepared at a width of 90 mm (single column) and 180 mm (double column) with a maximum height of 170mm.

colormap = cm.viridis
colors = colormap(np.linspace(0, 1, len(directories)))

plt.figure(figsize=(18.0 / 2.54, 18.0 / 2.54 * 2 / 3), dpi=300)  # 雙欄寬度，保持適合的比例

# 讀取並繪製每個掩膜比例的訓練損失
for (label, file_path), color in zip(directories.items(), colors):
    epochs = []
    train_losses = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data["epoch"])
            train_losses.append(data["train_loss"])
    
    plt.plot(epochs, train_losses, label=f"{label} Masking", color=color)

# 圖形設定
plt.xlabel("Epoch", fontsize=8, fontname="Arial")
plt.ylabel("Training Loss", fontsize=8, fontname="Arial")
plt.yscale("log")
#plt.title("Training Curves (Pre-trained on Circle Inclusion)", fontsize=7, fontname="Arial")
plt.legend(title="Masking Ratio (%)", fontsize=5, title_fontsize=7, loc='upper right')
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

# 保存為 PDF 和 TIFF 文件
output_dir = r"D:/Material_mask_autoencoder/results/pretrained_circle"
os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

plt.savefig(os.path.join(output_dir, "training_curve.pdf"), format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "training_curve.tiff"), format="tiff", dpi=300, bbox_inches='tight')

plt.show()
