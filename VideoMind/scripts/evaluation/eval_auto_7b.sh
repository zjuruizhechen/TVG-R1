#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

dataset=$1
split=$2

# model_gnd_path="Qwen/Qwen2.5-VL-7B-Instruct"
model_gnd_path=$3
model_name=$4
sys_prompt=$5
# model_name="Qwen2.5-VL-7B"

pred_path="outputs/${model_name}_${dataset}_${split}"

echo -e "\e[1;36mEvaluating:\e[0m $dataset ($split)"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_auto2.py \
        --dataset $dataset \
        --split $split \
        --pred_path $pred_path \
        --model_gnd_path $model_gnd_path \
        --sys_prompt $sys_prompt \
        --chunk $CHUNKS \
        --index $IDX &
done

wait

# python videomind/eval/eval_auto.py $pred_path
