@echo off
python D:\Material_mask_autoencoder\main_pretrain.py ^
    --data_path D:\2d_composite_mesh_generator\inclusion_train_100k ^
    --mask_ratio 0.90 ^
    --output_dir D:\Material_mask_autoencoder\output_dir\inclusion_mr090 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\inclusion_mr090

