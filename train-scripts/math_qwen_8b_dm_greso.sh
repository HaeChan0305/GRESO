set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# NCCL sane defaults for single-node multi-GPU on AWS
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1 # This should be needed for vLLM version 0.11.0
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# export NCCL_SOCKET_IFNAME=lo

export NCCL_NVLS_ENABLE=0
export VLLM_USE_SYMMETRIC_MEMORY=1
unset NCCL_SOCKET_IFNAME

# export VLLM_USE_V1='1'  # GRESO's verl does not support vLLM V1
export WANDB_API_KEY="79f4decc1667e5ef75c38f236c356ee5cc1c764b"
export WANDB_PROJECT="GRPO"
export WANDB_ENTITY="haechan-kaist"  # optional if using teams
export WANDB_MODE="online"  # or "offline", "disabled"
# export WANDB_RUN_ID="75h60ujw"
export HYDRA_FULL_ERROR=1
# export WANDB_RESUME='must'

experiment_name="qwen3-8b-greso-paper-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6-binreward"
save_path="./models/$experiment_name"
freq=15
rollout=8

cd /home/irteam/GRESO
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/irteam/MLILAB-GRPO/data/DAPO/train_17288.parquet \
    data.val_files=[/home/irteam/MLILAB-GRPO/data/MATH500/test.parquet,/home/irteam/MLILAB-GRPO/data/AIME2024/test.parquet,/home/irteam/MLILAB-GRPO/data/AIME2025/test.parquet,/home/irteam/MLILAB-GRPO/data/Minerva/test.parquet,/home/irteam/MLILAB-GRPO/data/AMC2024/test.parquet] \
    data.train_batch_size=128 \
    +data.dynamic_filtering=True \
    +data.dynamic_filtering_strategy=all_probabilistic \
    +data.p_easy=0.5 \
    +data.p_hard=0.5 \
    +data.target_zero_variance=0.25 \
    +data.sampling_batch_size=192 \
    +data.real_train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((1024 + 8192) * 4)) \
    +ttis.type=dynamic_sampling \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.n=$rollout \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=$rollout \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(((1024 + 8192) * 32)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_prev_epoch_qwen2_5_1_5b_MATH' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=$freq \
    trainer.default_local_dir=$save_path \
    trainer.test_freq=$freq \
    trainer.total_epochs=4 \
    +trainer.val_before_train=True $@
