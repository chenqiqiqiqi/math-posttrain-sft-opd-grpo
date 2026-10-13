#!/bin/bash
# D-SFT 第二步：用 verl 的 FSDP SFT 训练器（与 OPD/GRPO 同一代码库）在老师回答上做全参数 SFT。
# 超参数取论文表 3：学习率 1e-5、cosine、预热 5%、1 epoch、全局 batch 64；最长序列 = 1024 + 7168。
# 单轮格式：prompt 套聊天模板（含生成提示）+ 老师回答原文 + eos，思考内容完整保留。
#
# 用法：SFT_DATA=runs/sft_data/D/sft_data.parquet STUDENT=/root/autodl-tmp/models/S0 NGPUS=1 \
#       RUN_NAME=D-SFT bash scripts/train_sft.sh
set -euo pipefail
set -x

PB_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PB_ROOT/upstream/Rethinking-OPD/verl"
: "${SFT_DATA:?}"; : "${STUDENT:?}"; : "${RUN_NAME:?}"
NGPUS=${NGPUS:-1}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
MICRO=${MICRO:-1}
MAX_STEPS=${MAX_STEPS:-null}   # 试跑吞吐量测试时设一个小数
RUN_DIR=$PB_ROOT/runs/train/$RUN_NAME
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/train.log") 2>&1
export TENSORBOARD_DIR=$RUN_DIR/tensorboard PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
case "$SFT_DATA" in /*) ;; *) SFT_DATA=$PB_ROOT/$SFT_DATA ;; esac

{
  echo "date: $(date -Is)"; echo "student: $STUDENT"; echo "sft_data: $SFT_DATA"
  echo "sft_data_sha256: $(sha256sum "$SFT_DATA" | cut -d' ' -f1)"; echo "ngpus: $NGPUS"; echo "dtype: $MODEL_DTYPE"
  echo "git: $(git -C "$PB_ROOT" rev-parse HEAD 2>/dev/null || echo none)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$RUN_DIR/run_identity.txt"

T0=$(date +%s)
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$SFT_DATA" \
    data.val_files="$SFT_DATA" \
    data.val_max_samples=32 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.multiturn.enable=false \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=$MICRO \
    data.max_length=8192 \
    data.truncation=error \
    model.partial_pretrain="$STUDENT" \
    model.fsdp_config.model_dtype=$MODEL_DTYPE \
    model.enable_gradient_checkpointing=True \
    use_remove_padding=True \
    optim.lr=1e-5 \
    optim.lr_scheduler=cosine \
    optim.lr_warmup_steps_ratio=0.05 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.save_freq=${SAVE_FREQ:--1} \
    trainer.max_ckpt_to_keep=${MAX_CKPT_KEEP:-1} \
    trainer.test_freq=-1 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=project-b-opd \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    trainer.resume_mode=${RESUME_MODE:-auto} \
    "$@"
echo "{\"train_wall_sec\": $(( $(date +%s) - T0 )), \"ngpus\": $NGPUS}" >> "$RUN_DIR/sft_timing.jsonl"
