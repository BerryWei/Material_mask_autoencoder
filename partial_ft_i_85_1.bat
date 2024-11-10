set NUM_TAIL_BLOCKS=1
set NUM_DATASET=5000


python main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\inclusion_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr085/checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_partialfinetune_%NUM_TAIL_BLOCKS%_inclusion_mr085_cls_C00 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_partialfinetune_%NUM_TAIL_BLOCKS%_inclusion_mr085_cls_C00 ^
    --cls_token ^
	--partial_fine_tuning ^
	--num_tail_blocks %NUM_TAIL_BLOCKS% ^
    --numDataset %NUM_DATASET% ^
	--num_workers  2 ^
    --target_col_name C00 


python main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\inclusion_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr085/checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_partialfinetune_%NUM_TAIL_BLOCKS%_inclusion_mr085_cls_C11 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_partialfinetune_%NUM_TAIL_BLOCKS%_inclusion_mr085_cls_C11 ^
    --cls_token ^
	--partial_fine_tuning ^
	--num_tail_blocks %NUM_TAIL_BLOCKS% ^
    --numDataset %NUM_DATASET% ^
	--num_workers  2 ^
    --target_col_name C11

python main_finetune.py ^
    --data_path D:\2d_composite_mesh_generator\inclusion_valid ^
    --finetune D:\Material_mask_autoencoder\output_dir\inclusion_mr085/checkpoint-399.pth ^
    --output_dir D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_partialfinetune_%NUM_TAIL_BLOCKS%_inclusion_mr085_cls_C22 ^
    --log_dir    D:\Material_mask_autoencoder\output_dir\i2i_%NUM_DATASET%_partialfinetune_%NUM_TAIL_BLOCKS%_inclusion_mr085_cls_C22 ^
    --cls_token ^
	--partial_fine_tuning ^
	--num_tail_blocks %NUM_TAIL_BLOCKS% ^
    --numDataset %NUM_DATASET% ^
	--num_workers  2 ^
    --target_col_name C22

