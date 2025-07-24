set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant
 first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning
 process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
 <think> reasoning process here </think><answer> answer here </answer>"""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=leonardPKU/GEOQA_8K_R1V@train \
    data.val_files=leonardPKU/GEOQA_8K_R1V@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=r1v \
    trainer.experiment_name=qwen2_5_vl_3b_geoqa8k \
    trainer.n_gpus_per_node=8
