#!/bin/bash
#SBATCH --job-name=dapo          # 作业名称
#SBATCH --nodes=1                # 节点数
#SBATCH --ntasks-per-node=1      # 每个节点的任务数
#SBATCH --cpus-per-task=16       # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:4             # 需要 1 个 GPU
#SBATCH --time=24:00:00          # 最大运行时间
#SBATCH --mem=512G               # 内存大小
#SBATCH --output=logs/%j.log     # 输出日志路径
#SBATCH --partition=ghx4         # 指定分区
#SBATCH --account=bewc-dtai-gh   # 指定账户

CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

export WANDB_API_KEY=256879fdda25bc1fb8ee4f0310e71615e92f75c9
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1

unset ROCR_VISIBLE_DEVICES

HOME=$CACHE
MODEL_NAME=mistralai/Ministral-8B-Instruct-2410

ROLLOUT_N=8
PASS_K=1

LORA_RANK=0
LORA_ALPHA=128
LEARNING_RATE=3e-7
DATA_SEED=99
ROLLOUT_SEED=0
MICRO_BATCH_SIZE=8
TEMPERATURE=0.7
VAL_TEMPERATURE=0.7
SAVE_AND_TEST_INTERVAL=25

ADVANTAGE_ESTIMATOR=grpo
CORRECT_SAMPLE_LOG_PROB_COEF=0
INCORRECT_SAMPLE_LOG_PROB_COEF=-0.002

experiment_name=${ADVANTAGE_ESTIMATOR}_n${ROLLOUT_N}_k${PASS_K}_p${CORRECT_SAMPLE_LOG_PROB_COEF}_n${INCORRECT_SAMPLE_LOG_PROB_COEF}_seed${DATA_SEED}

echo "job is starting on `hostname`"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=[$HOME/verl-data/dapo14k/train.parquet] \
 data.val_files=[$HOME/verl-data/math/test.parquet,$HOME/verl-data/amc23/test.parquet,$HOME/verl-data/olympiadbench/test.parquet,$HOME/verl-data/aime24/test.parquet,$HOME/verl-data/aime25/test.parquet] \
 data.train_batch_size=128 \
 data.max_prompt_length=1024 \
 data.max_response_length=2048 \
 data.filter_overlong_prompts=True \
 data.chat_template_name=mistral-math \
 data.dataloader_num_workers=16 \
 data.truncation='error' \
 data.shuffle=True \
 data.seed=${DATA_SEED} \
 actor_rollout_ref.model.path=$HOME/hf_models/$MODEL_NAME \
 actor_rollout_ref.model.use_shm=True \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
 actor_rollout_ref.actor.clip_ratio_low=0.2 \
 actor_rollout_ref.actor.clip_ratio_high=0.25 \
 actor_rollout_ref.actor.loss_agg_mode=token-mean \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.000 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] \
 actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
 actor_rollout_ref.rollout.top_k=-1 \
 actor_rollout_ref.rollout.top_p=1 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.rollout.n=${ROLLOUT_N} \
 actor_rollout_ref.rollout.seed=${ROLLOUT_SEED} \
 actor_rollout_ref.rollout.layered_summon=True \
 actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
 actor_rollout_ref.rollout.val_kwargs.n=16 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
 actor_rollout_ref.model.lora_rank=${LORA_RANK} \
 actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
 actor_rollout_ref.rollout.load_format=safetensors \
 actor_rollout_ref.model.target_modules=all-linear \
 algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
 algorithm.pass_k=${PASS_K} \
 algorithm.use_kl_in_reward=False \
 algorithm.kl_ctrl.kl_coef=0.000 \
 algorithm.correct_sample_log_prob_coef=${CORRECT_SAMPLE_LOG_PROB_COEF} \
 algorithm.incorrect_sample_log_prob_coef=${INCORRECT_SAMPLE_LOG_PROB_COEF} \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=${SAVE_AND_TEST_INTERVAL} \
 trainer.max_actor_ckpt_to_keep=1 \
 trainer.test_freq=${SAVE_AND_TEST_INTERVAL} \
 trainer.logger=[console,wandb] \
 trainer.val_before_train=True \
 trainer.project_name=experiments-0909 \
 trainer.default_local_dir=/work/nvme/bdtp/gzeng/experiments-0909/${experiment_name} \
 trainer.experiment_name=${experiment_name} \
 trainer.total_epochs=3 \
 ray_init.temp_dir=$HOME/ray_tmp