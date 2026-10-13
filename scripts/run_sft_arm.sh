#!/bin/bash
# D-SFT 流水线（协议补充 4 第 1 条）：
#   m = floor(B_wall / 17366)；老师对全部 15,740 题各生成 m 条（双卡数据并行），保留未截断回答；
#   训练 1 epoch（约每 1/8 epoch 保存一次，后台导出 bf16、滚动保留 3 个）；终点导出后删除续训检查点。
#   生成与训练的墙钟都记入 sft_budget.json，与 B 对照如实报告。
# 用法：bash scripts/run_sft_arm.sh   （D-OPD 完成后）
set -u
cd "$(dirname "$0")/.."
source /root/pb-venv/bin/activate
M=/root/autodl-tmp/models
RUN=runs/train/D-SFT
DATA=runs/sft_data/D-SFT
mkdir -p "$RUN" "$DATA"
B=$(python -c "import json;print(json.load(open('runs/train/D-OPD/budget_ledger_final.json'))['spent_sec'])")
m=$(python -c "print(int($B // 17366))")
echo "B_wall=$B m=$m $(date -Is)" >> "$RUN/arm_summary.txt"
[ "$m" -ge 1 ] || { echo "ARM_FAILED m<1" >> "$RUN/arm_summary.txt"; exit 1; }

# 1) 老师生成（已完成则跳过，便于中断后重跑）
if [ ! -f "$DATA/meta.json" ]; then
  python scripts/sft_gen.py --teacher $M/T --n-prompts 15740 --samples-per-prompt "$m" --max-tokens 7168 \
    --gpu-ids 0,1 --out "$DATA" > "$RUN/sft_gen.log" 2>&1 || { echo "ARM_FAILED sft_gen $(date -Is)" >> "$RUN/arm_summary.txt"; exit 1; }
fi
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done; sleep 5
KEPT=$(python -c "import json;print(json.load(open('$DATA/meta.json'))['kept'])")
STEPS=$((KEPT / 64))
SAVE=$(( STEPS / 8 > 0 ? STEPS / 8 : 1 ))
echo "kept=$KEPT steps=$STEPS save_freq=$SAVE" >> "$RUN/arm_summary.txt"

# 2) SFT 训练 + 后台导出
rm -f "$RUN/TRAIN_DONE"
CUDA_VISIBLE_DEVICES= nohup python scripts/ckpt_exporter.py --run-dir "$RUN" --keep 3 --interval 60 \
  --stop-file "$RUN/TRAIN_DONE" >> "$RUN/exporter.log" 2>&1 &
EXP_PID=$!
T0=$(date +%s)
SFT_DATA=$DATA/sft_data.parquet STUDENT=$M/S0 NGPUS=2 MODEL_DTYPE=fp32 MICRO=1 SAVE_FREQ=$SAVE MAX_CKPT_KEEP=1 \
  RUN_NAME=D-SFT bash scripts/train_sft.sh
RC=$?
echo "{\"attempt_train_wall_sec\": $(( $(date +%s) - T0 )), \"rc\": $RC}" >> "$RUN/train_attempts.jsonl"
echo "$RC" > "$RUN/TRAIN_DONE"
wait $EXP_PID

FINAL=$(cat "$RUN/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)
python - "$DATA/meta.json" "$RUN/train_attempts.jsonl" "$B" > "$RUN/sft_budget.json" <<'EOF'
import json, sys
meta = json.load(open(sys.argv[1]))
att = [json.loads(l) for l in open(sys.argv[2])]
train_wall = sum(a["attempt_train_wall_sec"] for a in att if a["rc"] == 0) or att[-1]["attempt_train_wall_sec"]
gen_wall = meta["wall_sec_including_load"]
print(json.dumps({"B_wall": float(sys.argv[3]), "gen_wall": gen_wall, "train_wall_ok_attempts": train_wall,
                  "total_wall": gen_wall + train_wall, "ratio_to_B": (gen_wall + train_wall) / float(sys.argv[3]),
                  "kept": meta["kept"], "generated": meta["generated"], "truncation_rate": meta["truncation_rate"],
                  "attempts": att}, indent=1))
EOF
if [ "$RC" = 0 ] && [ -f "$RUN/hf/step_$FINAL/model.safetensors" ]; then
  du -sh "$RUN/ckpt" >> "$RUN/arm_summary.txt"
  rm -rf "$RUN/ckpt/global_step_"*        # 协议补充 2 第 6 条：终点已导出，续训检查点不再需要
  echo "ARM_OK final_step=$FINAL $(date -Is)" >> "$RUN/arm_summary.txt"
else
  echo "ARM_FAILED rc=$RC final_step=$FINAL $(date -Is)（未删除任何检查点）" >> "$RUN/arm_summary.txt"
fi
cat "$RUN/arm_summary.txt"
