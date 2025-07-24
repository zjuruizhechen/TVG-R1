#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

dataset=$1
split=$2
model_gnd_path=$3
model_name=$4
sys_prompt=$5

pred_path="outputs/${model_name}_${dataset}_${split}"

echo -e "\e[1;36mEvaluating:\e[0m $dataset ($split)"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
TOTAL_GPUS=${#GPULIST[@]}
GROUP_SIZE=2
NUM_GROUPS=$((TOTAL_GPUS / GROUP_SIZE))

for IDX in $(seq 0 $((NUM_GROUPS-1))); do
    GPU0=${GPULIST[$((IDX * GROUP_SIZE))]}
    GPU1=${GPULIST[$((IDX * GROUP_SIZE + 1))]}
    VISIBLE_GPUS="$GPU0,$GPU1"

    echo -e "\e[1;33mLaunching group $IDX on GPUs: $VISIBLE_GPUS\e[0m"

    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_GPUS python videomind/eval/infer_auto2.py \
        --dataset $dataset \
        --split $split \
        --pred_path $pred_path \
        --model_gnd_path $model_gnd_path \
        --sys_prompt $sys_prompt \
        --chunk $NUM_GROUPS \
        --index $IDX &
done

wait

# python videomind/eval/eval_auto.py $pred_path
