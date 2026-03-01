set -x

export PYTHONPATH=$PYTHONPATH:$(dirname "$0")/..
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 设置数据和模型路径 (假设 AutoDL 上也是这个相对路径结构，或者你可以修改为绝对路径)
# 注意：在 Linux 上路径分隔符是 /
BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_FILES="$BASE_DIR/dataset-verl/train/data.parquet"
VAL_FILES="$BASE_DIR/dataset-verl/valid/data.parquet"
MODEL_PATH="$BASE_DIR/pretrained_model"
OUTPUT_DIR="$BASE_DIR/outputs"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard_log"

python3 - <<'PY'
try:
    from transformers import AutoModelForVision2Seq
except Exception as e:
    print("Transformers 版本过低或环境缺少 AutoModelForVision2Seq，请先升级 transformers。")
    raise
PY

python3 - <<'PY'
try:
    import qwen_vl_utils
except Exception as e:
    print("缺少 qwen_vl_utils，请先安装 qwen_vl_utils。")
    raise
PY

cd "$BASE_DIR/dataset-verl"

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=$OUTPUT_DIR \
    hydra.output_subdir=null \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=32 \
    +data.gen_batch_size=4 \
    data.max_prompt_length=1024 \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=4 \
    data.val_batch_size=30 \
    +data.val_subset_size=200 \
    +data.val_subset_seed=1 \
    data.truncation=left \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    reward_model.reward_manager=my_reward_manager \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=700 \
    trainer.resume_mode=auto \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    +trainer.save_best_ckpt=True \
    +trainer.best_ckpt_higher_is_better=True \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.val_before_train=False \
    trainer.project_name='math-pro-grpo' \
    trainer.experiment_name='qwen2.5-vl-sft-verl' \
    trainer.logger='["tensorboard","console"]' $@
