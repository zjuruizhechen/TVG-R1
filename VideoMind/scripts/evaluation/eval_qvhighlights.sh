#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

model_gnd_path="model_zoo/VideoMind-2B-FT-QVHighlights"

pred_path="outputs/qvhighlights_ft_val"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_qvhighlights.py \
        --dataset qvhighlights \
        --split valid \
        --pred_path $pred_path \
        --model_gnd_path $model_gnd_path \
        --chunk $CHUNKS \
        --index $IDX &
done

wait

cat $pred_path/*.jsonl > $pred_path/hl_val_submission.jsonl

python videomind/eval/eval_qvhighlights.py $pred_path/hl_val_submission.jsonl

# ==================== test split ====================

# pred_path="outputs/qvhighlights_ft_test"

# IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
# CHUNKS=${#GPULIST[@]}

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_qvhighlights.py \
#         --dataset qvhighlights \
#         --split test \
#         --pred_path $pred_path \
#         --model_gnd_path $model_gnd_path \
#         --chunk $CHUNKS \
#         --index $IDX &
# done

# wait

# cat $pred_path/*.jsonl > $pred_path/hl_test_submission.jsonl
