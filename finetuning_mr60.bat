

set NUM_DATASET=5000

python D:\Material_mask_autoencoder\main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr060\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2c_%NUM_DATASET%_finetune_inclusion_mr060_cls_C00 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2c_%NUM_DATASET%_finetune_inclusion_mr060_cls_C00 ^
    --cls_token ^
    --numDataset %NUM_DATASET% ^
    --target_col_name C00

python D:\Material_mask_autoencoder\main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr060\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2c_%NUM_DATASET%_finetune_inclusion_mr060_cls_C11 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2c_%NUM_DATASET%_finetune_inclusion_mr060_cls_C11 ^
    --cls_token ^
    --numDataset %NUM_DATASET% ^
    --target_col_name C11

python D:\Material_mask_autoencoder\main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\circle_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr060\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2c_%NUM_DATASET%_finetune_inclusion_mr060_cls_C22 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2c_%NUM_DATASET%_finetune_inclusion_mr060_cls_C22 ^
    --cls_token ^
    --numDataset %NUM_DATASET% ^
    --target_col_name C22




python D:\Material_mask_autoencoder\main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\inclusion_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr060\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_finetune_inclusion_mr060_cls_C00 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_finetune_inclusion_mr060_cls_C00 ^
    --cls_token ^
    --numDataset %NUM_DATASET% ^
    --target_col_name C00

python D:\Material_mask_autoencoder\main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\inclusion_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr060\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_finetune_inclusion_mr060_cls_C11 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_finetune_inclusion_mr060_cls_C11 ^
    --cls_token ^
    --numDataset %NUM_DATASET% ^
    --target_col_name C11

python D:\Material_mask_autoencoder\main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\inclusion_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr060\checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_finetune_inclusion_mr060_cls_C22 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_finetune_inclusion_mr060_cls_C22 ^
    --cls_token ^
    --numDataset %NUM_DATASET% ^
    --target_col_name C22