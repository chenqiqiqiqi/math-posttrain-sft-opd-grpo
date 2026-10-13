#!/bin/bash
# 项目 B 的 verl 训练启动脚本（OPD / GRPO），基于上游 on_policy_distillation.sh 与 grpo.sh。
# 与上游的差别只在：数据路径、卡数、精度、步数上限、日志后端、墙钟预算停止（PB_BUDGET_SEC）。
#
# 用法（在服务器上）：
#   ARM=D-OPD STUDENT=/root/autodl-tmp/models/S0 TEACHER=/root/autodl-tmp/models/T \
#   NGPUS=1 MODEL_DTYPE=fp32 MAX_STEPS=20 PB_BUDGET_SEC=0 RUN_NAME=pilot-dopd \
#   bash scripts/train_verl.sh
#
# ARM: D-OPD | D-GRPO | Q-OPD（学生起点由 STUDENT 决定，Q-OPD 与 Q-SFT→OPD 共用）
set -euo pipefail
set -x

PB_ROOT=$(cd "$(dirname "$0")/.." && pwd)
UP=$PB_ROOT/upstream/Rethinking-OPD
cd "$UP"

: "${ARM:?需要 ARM}"; : "${STUDENT:?需要 STUDENT}"; : "${RUN_NAME:?需要 RUN_NAME}"
NGPUS=${NGPUS:-1}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}          # 试跑 P3 决定 fp32 / bfloat16
MAX_STEPS=${MAX_STEPS:-1000}              # 安全上限；正式实验由 PB_BUDGET_SEC 截停
export PB_BUDGET_SEC=${PB_BUDGET_SEC:-0}  # 墙钟预算秒数（0 = 不启用）
MAX_PROMPT_LENGTH=1024
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
BATCH=64
SAVE_FREQ=${SAVE_FREQ:-20}
SEED=${SEED:-42}

case "$ARM" in
  D-OPD)  ADV=token_reward_direct; N=4; TOPK=16; TEACH=1; DATA=$PB_ROOT/data/prepared/train_dapo.parquet; THINK="" ;;
  D-GRPO) ADV=grpo;                N=8; TOPK=0;  TEACH=0; DATA=$PB_ROOT/data/prepared/train_dapo.parquet; THINK="" ;;
  Q-OPD)  ADV=token_reward_direct; N=4; TOPK=16; TEACH=1; DATA=$PB_ROOT/data/prepared/train_ot3.parquet
          THINK="+data.apply_chat_template_kwargs.enable_thinking=False" ;;
  *) echo "未知 ARM: $ARM"; exit 1 ;;
esac
if [ "$TEACH" = 1 ]; then : "${TEACHER:?OPD 需要 TEACHER}"; fi

RUN_DIR=$PB_ROOT/runs/train/$RUN_NAME
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/train.log") 2>&1
export TENSORBOARD_DIR=$RUN_DIR/tensorboard
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 RAY_memory_usage_threshold=0.99
export VLLM_USE_FLASHINFER_SAMPLER=0   # 服务器 CUDA 13.0 工具包缺 curand.h，FlashInfer 采样无法编译；PyTorch 原生采样分布相同
# 每张卡单次前向/反向的 token 上限：只影响显存与速度，不改变算法（梯度在 mini-batch 内累积）。
# 上游按 80GB 设 32768；48GB 卡建议 8192。下限是一条完整序列（prompt + 最长回答）。
MIN_TOK=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
[ "$PPO_MAX_TOKEN_LEN_PER_GPU" -lt "$MIN_TOK" ] && PPO_MAX_TOKEN_LEN_PER_GPU=$MIN_TOK
# UPDATE_TOKLEN：学生参数更新（需反向传播）每张卡单次 token 上限；PPO_MAX_TOKEN_LEN_PER_GPU 只用于前向计算概率。
# 2026-10-08 D-OPD 第 2 步在 16384 下反向传播 OOM，更新改用 8192（协议补充 5）。
UPDATE_TOKLEN=${UPDATE_TOKLEN:-$PPO_MAX_TOKEN_LEN_PER_GPU}
[ "$UPDATE_TOKLEN" -lt "$MIN_TOK" ] && UPDATE_TOKLEN=$MIN_TOK
RM_MICRO=${RM_MICRO:-24}   # 老师前向每次的序列数，显存不够时调小
# OFFLOAD=1：学生参数/优化器状态与老师参数在不用时卸载到 CPU 内存（48GB 卡 + fp32 必需）。只影响显存与速度，不改变算法。
OFFLOAD=${OFFLOAD:-0}
if [ "$OFFLOAD" = 1 ]; then ACT_OFF=True; else ACT_OFF=False; fi
MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))

ARM_ARGS=()
if [ "$TEACH" = 1 ]; then
  ARM_ARGS+=(
    reward_model.enable=True
    reward_model.model.path="$TEACHER"
    reward_model.model.input_tokenizer=null
    reward_model.model.use_remove_padding=True
    reward_model.model.fsdp_config.param_offload=$ACT_OFF
    +reward_model.model.dtype=$MODEL_DTYPE
    reward_model.micro_batch_size_per_gpu=$RM_MICRO
    +actor_rollout_ref.rollout.top_k_strategy=only_stu
    +actor_rollout_ref.rollout.reward_weight_mode=student_p
    +actor_rollout_ref.rollout.teacher_temperature=1.0
    trainer.is_plot=True
  )
else
  ARM_ARGS+=(
    reward_model.enable=False
    ++algorithm.rollout_correction.rollout_is=token
    ++algorithm.rollout_correction.rollout_is_threshold=2.0
    trainer.is_plot=False
  )
fi
[ -n "$THINK" ] && ARM_ARGS+=("$THINK")

# 记录本次运行的身份信息
{
  echo "date: $(date -Is)"; echo "arm: $ARM"; echo "student: $STUDENT"; echo "teacher: ${TEACHER:-}"
  echo "ngpus: $NGPUS"; echo "dtype: $MODEL_DTYPE"; echo "max_steps: $MAX_STEPS"; echo "budget_sec: $PB_BUDGET_SEC"
  echo "ppo_max_token_len_per_gpu (forward): $PPO_MAX_TOKEN_LEN_PER_GPU"; echo "update_toklen: $UPDATE_TOKLEN"; echo "rm_micro: ${RM_MICRO}"; echo "offload: $OFFLOAD"; echo "gpu_mem_util: ${GPU_MEM_UTIL:-0.6}"; echo "data: $DATA"; echo "data_sha256: $(sha256sum "$DATA" | cut -d' ' -f1)"; echo "seed: $SEED"
  echo "git: $(git -C "$PB_ROOT" rev-parse HEAD 2>/dev/null || echo none)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$RUN_DIR/run_identity.txt"

ray stop --force || true
ray start --head --num-gpus="$NGPUS"
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV \
    algorithm.grpo_outcome_weight=1.0 \
    data.shuffle=False \
    data.seed=$SEED \
    data.train_files="['$DATA']" \
    data.val_files="['$PB_ROOT/data/prepared/dev_deepmath400.parquet']" \
    data.train_batch_size=$BATCH \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$STUDENT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$UPDATE_TOKLEN \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACT_OFF \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACT_OFF \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    +actor_rollout_ref.rollout.log_prob_top_k=$TOPK \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +reward_model.reward_kwargs.enable_format_reward=False \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=project-b-opd \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.total_epochs=3 \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-1} \
    "${ARM_ARGS[@]}" \
    "$@"
