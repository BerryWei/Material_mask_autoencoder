@echo off
python D:\Material_mask_autoencoder\main_linprobe.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\circle_mr075\checkpoint-799.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\linprobe_circular_mse_gp ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\linprobe_circular_mse_gp ^
    --global_pool

python D:\Material_mask_autoencoder\main_linprobe.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\circle_mr075\checkpoint-799.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\linprobe_circular_mse_cls ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\linprobe_circular_mse_cls ^
    --cls_token