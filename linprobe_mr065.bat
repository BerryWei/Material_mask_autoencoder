@echo off

python D:\Material_mask_autoencoder\main_linprobe.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\circle_mr065\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_Vf_real ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_Vf_real ^
    --cls_token ^
    --target_col_name Vf_real

python D:\Material_mask_autoencoder\main_linprobe.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\circle_mr065\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_C00 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_C00 ^
    --cls_token ^
    --target_col_name C00

python D:\Material_mask_autoencoder\main_linprobe.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\circle_mr065\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_C11 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_C11 ^
    --cls_token ^
    --target_col_name C11

python D:\Material_mask_autoencoder\main_linprobe.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\circle_mr065\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_C22 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr065_cls_C22 ^
    --cls_token ^
    --target_col_name C22

