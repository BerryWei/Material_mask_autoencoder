@echo off
python D:\Material_mask_autoencoder\main_pretrain.py ^
    --data_path D:\2d_composite_mesh_generator\circle ^
    --mask_ratio 0.75 ^
    --output_dir D:\Material_mask_autoencoder\output_dir\circle_mr075 ^
    --log_dir D:\Material_mask_autoencoder\output_dir\circle_mr075


python D:\Material_mask_autoencoder\main_pretrain.py ^
    --data_path D:\2d_composite_mesh_generator\circle ^
    --mask_ratio 0.7 ^
    --output_dir D:\Material_mask_autoencoder\output_dir\circle_mr070 ^
    --log_dir D:\Material_mask_autoencoder\output_dir\circle_mr070

python D:\Material_mask_autoencoder\main_pretrain.py ^
    --data_path D:\2d_composite_mesh_generator\circle ^
    --mask_ratio 0.75 ^
    --output_dir D:\Material_mask_autoencoder\output_dir\circle_mr075 ^
    --log_dir D:\Material_mask_autoencoder\output_dir\circle_mr075


python D:\Material_mask_autoencoder\main_pretrain.py ^
    --data_path D:\2d_composite_mesh_generator\circle ^
    --mask_ratio 0.65 ^
    --output_dir D:\Material_mask_autoencoder\output_dir\circle_mr065 ^
    --log_dir D:\Material_mask_autoencoder\output_dir\circle_mr065