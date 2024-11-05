#!/bin/bash 
#SBATCH --account=GOV113037       # (-A) iService Project ID
#SBATCH --job-name=mae_10_sfc
#SBATCH --partition=gp4d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=berrya90239@gmail.com


conda init
conda activate timmcuda121


CUDA_VISIBLE_DEVICES=0 python main_pretrain.py \
    --data_path /work/berrya90239/Material_mask_autoencoder/inclusion_train_100k \
	--batch_size 100 \
	--epochs 400 \
    --mask_ratio 0.10 \
    --output_dir /work/berrya90239/Material_mask_autoencoder/output_dir/inclusion_mr010 \
    --log_dir    /work/berrya90239/Material_mask_autoencoder/output_dir/inclusion_mr010 \
	--num_workers 2 & 
	

CUDA_VISIBLE_DEVICES=1 python main_pretrain.py \
    --data_path /work/berrya90239/Material_mask_autoencoder/inclusion_train_100k \
	--batch_size 100 \
	--epochs 400 \
    --mask_ratio 0.15 \
    --output_dir /work/berrya90239/Material_mask_autoencoder/output_dir/inclusion_mr015 \
    --log_dir    /work/berrya90239/Material_mask_autoencoder/output_dir/inclusion_mr015 \
	--num_workers 2 & 
	
wait