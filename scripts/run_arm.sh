#!/bin/bash
# 正式实验的单组流水线（verl 组：D-OPD / D-GRPO / Q-OPD）。
#   训练（train_verl.sh）+ 后台 ckpt_exporter（bf16 HF，滚动保留 3 个）
#   → 训练成功且终点已导出 → 按协议补充 2 第 6 条删除续训检查点。
#   训练失败：不删除任何东西；用同一 RUN_NAME 重跑即从最近检查点恢复（台账同步恢复）。
# 用法：ARM=D-OPD STUDENT=... TEACHER=... RUN_NAME=D-OPD MAX_STEPS=150 [PB_BUDGET_SEC=...] bash scripts/run_arm.sh
set -u
cd "$(dirname "$0")/.."
source /root/pb-venv/bin/activate
: "${RUN_NAME:?}"
RUN=runs/train/$RUN_NAME
mkdir -p "$RUN"
rm -f "$RUN/TRAIN_DONE"
# 选定的速度设置（补充 3 第 2 条，速度测试 V2）
export NGPUS=${NGPUS:-2} MODEL_DTYPE=fp32 OFFLOAD=1 PPO_MAX_TOKEN_LEN_PER_GPU=16384 UPDATE_TOKLEN=8192 GPU_MEM_UTIL=0.7 RM_MICRO=8
export SAVE_FREQ=${SAVE_FREQ:-20} MAX_CKPT_KEEP=1

CUDA_VISIBLE_DEVICES= nohup python scripts/ckpt_exporter.py --run-dir "$RUN" --keep 3 --interval 60 \
  --stop-file "$RUN/TRAIN_DONE" >> "$RUN/exporter.log" 2>&1 &
EXP_PID=$!

bash scripts/train_verl.sh
RC=$?
echo "$RC" > "$RUN/TRAIN_DONE"
ray stop --force >/dev/null 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done
wait $EXP_PID

FINAL=$(cat "$RUN/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)
if [ "$RC" = 0 ] && [ -f "$RUN/hf/step_$FINAL/model.safetensors" ]; then
  du -sh "$RUN/ckpt" >> "$RUN/arm_summary.txt"
  rm -rf "$RUN/ckpt/global_step_"*        # 协议补充 2 第 6 条：终点已导出，续训检查点不再需要
  echo "ARM_OK final_step=$FINAL $(date -Is)" >> "$RUN/arm_summary.txt"
else
  echo "ARM_FAILED rc=$RC final_step=$FINAL $(date -Is)（未删除任何检查点）" >> "$RUN/arm_summary.txt"
fi
cp "$RUN/ckpt/budget_ledger.json" "$RUN/budget_ledger_final.json" 2>/dev/null
cat "$RUN/arm_summary.txt"
