set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=RuizheChen/TVG_Coldstart

SYSTEM_PROMPT="""You FIRST think about the reasoning process in the mind and finally determine the precise time period related to the query. 
  The reasoning process MUST BE enclosed within <think> </think> tags. The specific time period MUST BE in the format [start time, end time] in seconds enclosed within <time> </time> tags. For example, <think>the reasoning process</think> <time>[5.2, 10.4]</time> """

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=examples/data_config/TVG_RL_train.yaml \
    data.val_files=examples/data_config/TVG_RL_val.yaml \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.rollout_batch_size=256 \
    worker.actor.global_batch_size=64 \
    worker.actor.entropy_coeff=1e-3 \
    worker.actor.kl_loss_coef=0.0 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.compute_score=tvgonly \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=RL_Temporal_Grounded_QA_Grounding_filtered_ColdStart \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=100 \
    trainer.val_before_train=true \
    trainer.logger=[\"console\",\"wandb\"] \
    trainer.save_checkpoint_path=./models/RL_Temporal_Grounded_QA_Grounding_filtered_ColdStart \
    trainer.load_checkpoint_path=./models/RL_Temporal_Grounded_QA_Grounding_filtered_ColdStart \
    data.min_pixels=3136 \
    data.max_pixels=1605632 \
    data.nframes=64 \
    data.system_prompt="${SYSTEM_PROMPT}" \
    trainer.total_episodes=20 \
    trainer.val_freq=20 \


# python /opt/tiger/video-r1/aaa.py
