@echo off

python D:\Material_mask_autoencoder\main_linprobe.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid_test\ ^
    --finetune D:\Material_mask_autoencoder\output_dir\circle_mr075\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr075_cls_Vf_real_woBN_wDataAug_invDataset ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\linprobe_circular_mr075_cls_Vf_real_woBN_wDataAug_invDataset ^
    --cls_token ^
    --target_col_name Vf_real



