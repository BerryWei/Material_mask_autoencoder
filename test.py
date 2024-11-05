import os

# 目標目錄
base_dir = r"D:\Material_mask_autoencoder\output_dir"

# 遍歷該目錄中的所有文件夾
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # 檢查是否為目標格式的文件夾
    if os.path.isdir(folder_path) and folder_name.startswith("c2c_"):
        # 新的文件夾名稱
        new_folder_name = f"c2c_5000_{folder_name[13:]}"
        new_folder_path = os.path.join(base_dir, new_folder_name)
        
        # 重命名文件夾
        os.rename(folder_path, new_folder_path)
        print(f"Renamed: {folder_name} -> {new_folder_name}")

print("重命名完成。")
