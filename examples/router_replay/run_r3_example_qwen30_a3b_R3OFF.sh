#!/usr/bin/env bash
# Copy of verl/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh with:
#   - HF_MODEL_PATH / TRAIN_DATA_PATH / TEST_DATA_PATH placeholders filled
#   - total_training_steps=1 (user directive)
#   - test_freq=-1 (skip eval, fastest path)
# All R3 plumbing (ROUTING_REPLAY_MODE=R3, ENABLE_ROLLOUT_ROUTING_REPLAY=True,
# moe_enable_deepep=True, moe_token_dispatcher_type=flex, etc.) kept as-is.
#
# Run inside container vllm-r3-repro (has pip install -e /vllm-src/vllm-r3rfc).

set -x

NODES=1

# R3-OFF baseline: NO routing replay at all (used for sanity that the bug requires R3)
ROUTING_REPLAY_MODE=""

if [ "$ROUTING_REPLAY_MODE" = "R3" ]; then
    ENABLE_ROLLOUT_ROUTING_REPLAY=True
else
    ENABLE_ROLLOUT_ROUTING_REPLAY=False
fi

# Point HF to the shared cache where we've downloaded Qwen3-30B-A3B
export HF_HOME=/mnt/shared/hf-models
export HF_HUB_CACHE=/mnt/shared/hf-models

DIST_CKPT_PATH=""
HF_MODEL_PATH=Qwen/Qwen3-30B-A3B
TRAIN_DATA_PATH=/root/data/gsm8k/train.parquet
TEST_DATA_PATH=/root/data/gsm8k/test.parquet

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
PP=1
VPP=None
TP=2
EP=8
ETP=1
VLLM_INFER_TP=2
offload=True
gpu_memory_utilization=0.65
bs=8
micro_bs=3
use_dynamic_bsz=True
max_prompt_length=1024
max_response_length=1024
ppo_mini_batch_size=8
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

USE_LEGACY_WORKER_IMPL="disable" # disable, enable
if [ "$USE_LEGACY_WORKER_IMPL" = "disable" ]; then
    ROUTING_REPLAY_MODE_ARG="actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTING_REPLAY_MODE}"
    remove_padding=True
else
    ROUTING_REPLAY_MODE_ARG="actor_rollout_ref.actor.router_replay.mode=${ROUTING_REPLAY_MODE}"
    remove_padding=False
fi
exper_name=R3OFF_Node${NODES}_bs${bs}_${PP}${TP}${EP}${ETP}_${VLLM_INFER_TP}_minbs${ppo_mini_batch_size}_micro_bs${micro_bs}

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    data.train_batch_size=$bs \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=$remove_padding \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_INFER_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_r3_proof' \
    trainer.experiment_name="$exper_name" \
    trainer.nnodes=$NODES \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.balance_batch=False \
    trainer.use_legacy_worker_impl=${USE_LEGACY_WORKER_IMPL} \
    trainer.val_before_train=False 2>&1
