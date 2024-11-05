

if [ -z "$MR_VALUE" ] || [ -z "$NUM_DATASET" ]; then
    echo "Error: MR_VALUE and NUM_DATASET must be provided."
    exit 1
fi

conda init
conda activate timmcuda121

# 定義變數
FINETUNE_PATH="/work/berrya90239/Material_mask_autoencoder/output_dir/inclusion_mr${MR_VALUE}/checkpoint-399.pth"
DATA_PATH_INCLUSION="/work/berrya90239/Material_mask_autoencoder/inclusion_valid"
DATA_PATH_CIRCLE="/work/berrya90239/Material_mask_autoencoder/circle_valid"
COMMON_ARGS="--cls_token --num_workers 1 --numDataset ${NUM_DATASET}"

# 第一組任務：Inclusion Data Path
TARGETS=("Vf_real" "C00" "C11" "C22")
for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i python main_linprobe.py \
        --data_path ${DATA_PATH_INCLUSION} \
        --finetune ${FINETUNE_PATH} \
        --output_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2i_${NUM_DATASET}_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        --log_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2i_${NUM_DATASET}_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        ${COMMON_ARGS} \
        --target_col_name ${TARGETS[i]} &
done

wait

for i in 2 3; do
    CUDA_VISIBLE_DEVICES=$((i - 2)) python main_linprobe.py \
        --data_path ${DATA_PATH_INCLUSION} \
        --finetune ${FINETUNE_PATH} \
        --output_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2i_${NUM_DATASET}_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        --log_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2i_${NUM_DATASET}_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        ${COMMON_ARGS} \
        --target_col_name ${TARGETS[i]} &
done

wait

# 第二組任務：Circle Data Path
for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i python main_linprobe.py \
        --data_path ${DATA_PATH_CIRCLE} \
        --finetune ${FINETUNE_PATH} \
        --output_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2c_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        --log_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2c_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        ${COMMON_ARGS} \
        --target_col_name ${TARGETS[i]} &
done

wait

for i in 2 3; do
    CUDA_VISIBLE_DEVICES=$((i - 2)) python main_linprobe.py \
        --data_path ${DATA_PATH_CIRCLE} \
        --finetune ${FINETUNE_PATH} \
        --output_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2c_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        --log_dir /work/berrya90239/Material_mask_autoencoder/output_dir/i2c_linprobe_inclusion_mr${MR_VALUE}_cls_${TARGETS[i]} \
        ${COMMON_ARGS} \
        --target_col_name ${TARGETS[i]} &
done

wait
