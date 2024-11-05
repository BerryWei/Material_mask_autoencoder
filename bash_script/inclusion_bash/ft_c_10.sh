#!/bin/bash 
#SBATCH --account=GOV113037         # iService Project ID
#SBATCH --job-name=ft_10
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
#SBATCH --export=ALL

# 初始化 Conda 環境
conda init
conda activate timmcuda121

# 定義 MR_VALUE 和 NUM_DATASET 的變化範圍
MR_VALUES=(10 15)
NUM_DATASETS=(5000)

# 遍歷所有組合並執行任務
for MR_VALUE in "${MR_VALUES[@]}"; do
    for NUM_DATASET in "${NUM_DATASETS[@]}"; do
        export MR_VALUE=${MR_VALUE}
        export NUM_DATASET=${NUM_DATASET}
        bash run_finetune.sh
    done
done
