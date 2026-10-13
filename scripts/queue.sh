#!/bin/bash
# 服务器端自动排队（协议补充 4 第 3 条顺序 + 补充 6），不依赖本地会话：
#   等 D-OPD → B → [D-OPD 开发集评测 → 只留终点导出] → D-GRPO → [评测+清理] → D-SFT → [评测+清理]
#   → Q-OPD → [评测+清理] → Q-SFT-OPD → [评测+清理] → 基线开发集评测 → 测试集评测（所有模型冻结后，只此一次）→ 汇总
#   每组训练失败时自动重跑一次（从最近检查点恢复）；仍失败则停止排队等人工处理。
#   每组开始前检查数据盘可用空间是否满足该组峰值需求，不足则等待（不删除任何东西）。
# 用法：setsid nohup bash scripts/queue.sh > runs/queue.log 2>&1 &
cd "$(dirname "$0")/.."
source /root/pb-venv/bin/activate
M=/root/autodl-tmp/models
Q=runs/queue_status.txt
log() { echo "$(date -Is) $*" | tee -a $Q; }
last_status() { tail -1 "runs/train/$1/arm_summary.txt" 2>/dev/null | grep -oE "ARM_OK|ARM_FAILED"; }
wait_disk() {  # wait_disk <名称> <所需 GB>
  while [ "$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc 0-9)" -lt "$2" ]; do
    log "WAITING_DISK need=$2G free=$(df -h --output=avail /root/autodl-tmp | tail -1) before $1"; sleep 600
  done
}
cleanup_gpu() { ray stop --force >/dev/null 2>&1; sleep 3; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done; sleep 5; }
run_with_retry() {  # run_with_retry <名称> <所需 GB> <命令…>
  local name=$1 need=$2; shift 2
  for attempt in 1 2; do
    wait_disk "$name" "$need"; log "START $name attempt=$attempt"
    "$@" > "runs/train/$name/arm_driver_$attempt.log" 2>&1
    cleanup_gpu
    if [ "$(last_status "$name")" = ARM_OK ]; then log "DONE $name"; return 0; fi
    log "FAILED $name attempt=$attempt"
  done
  log "QUEUE_STOPPED at $name（需人工处理）"; exit 1
}
# 补充 6：该组 3 个导出在开发集评完（graded.csv 齐全）后，只保留终点（最大步数）导出
dev_and_prune() {
  local arm=$1
  log "DEV_EVAL $arm"
  python scripts/eval_jobs.py --plan dev --only "$arm@" > "runs/eval_dev_$arm.log" 2>&1
  cleanup_gpu
  local steps; steps=$(ls -d runs/train/$arm/hf/step_* 2>/dev/null | sed 's/.*step_//' | sort -n)
  local final; final=$(echo "$steps" | tail -1)
  for s in $steps; do
    if [ ! -f "runs/eval/dev/$arm@$s/graded.csv" ]; then log "DEV_EVAL_INCOMPLETE $arm@$s（不清理）"; return 0; fi
  done
  for s in $steps; do [ "$s" != "$final" ] && rm -rf "runs/train/$arm/hf/step_$s"; done
  log "DEV_DONE $arm kept=step_$final"
}

# 1) 等 D-OPD
log "QUEUE_START (v2) waiting for D-OPD"
until [ -n "$(last_status D-OPD)" ]; do sleep 120; done
if [ "$(last_status D-OPD)" != ARM_OK ]; then
  run_with_retry D-OPD 52 env ARM=D-OPD STUDENT=$M/S0 TEACHER=$M/T RUN_NAME=D-OPD MAX_STEPS=150 SAVE_FREQ=20 PB_BUDGET_SEC=0 bash scripts/run_arm.sh
fi
cleanup_gpu
B=$(python -c "import json;print(json.load(open('runs/train/D-OPD/budget_ledger_final.json'))['spent_sec'])")
echo "{\"B_wall_sec\": $B, \"ngpus\": 2, \"B_gpu_hours\": $(python -c "print(round($B*2/3600,3))"), \"source\": \"runs/train/D-OPD/budget_ledger_final.json\", \"time\": \"$(date -Is)\"}" > runs/B.json
log "B_wall=$B"
dev_and_prune D-OPD

# 2) D-GRPO
mkdir -p runs/train/D-GRPO
run_with_retry D-GRPO 52 env ARM=D-GRPO STUDENT=$M/S0 RUN_NAME=D-GRPO MAX_STEPS=1000 SAVE_FREQ=20 PB_BUDGET_SEC=$B bash scripts/run_arm.sh
dev_and_prune D-GRPO
# 3) D-SFT
mkdir -p runs/train/D-SFT
run_with_retry D-SFT 64 bash scripts/run_sft_arm.sh
dev_and_prune D-SFT
# 4) Q 族（等步数，各 100 步）
mkdir -p runs/train/Q-OPD runs/train/Q-SFT-OPD
run_with_retry Q-OPD 66 env ARM=Q-OPD STUDENT=$M/QB TEACHER=$M/QT RUN_NAME=Q-OPD MAX_STEPS=100 SAVE_FREQ=20 PB_BUDGET_SEC=0 bash scripts/run_arm.sh
dev_and_prune Q-OPD
run_with_retry Q-SFT-OPD 66 env ARM=Q-OPD STUDENT=$M/QS TEACHER=$M/QT RUN_NAME=Q-SFT-OPD MAX_STEPS=100 SAVE_FREQ=20 PB_BUDGET_SEC=0 bash scripts/run_arm.sh
dev_and_prune Q-SFT-OPD
log "TRAINING_DONE all arms"

# 5) 基线开发集评测；6) 测试集评测（所有模型已冻结，只此一次）；7) 汇总
log "DEV_EVAL baselines"
python scripts/eval_jobs.py --plan dev > runs/eval_dev_rest.log 2>&1; cleanup_gpu
log "TEST_EVAL start (all models frozen)"
python scripts/eval_jobs.py --plan test > runs/eval_test.log 2>&1; cleanup_gpu
log "TEST_EVAL done"
python scripts/report.py > runs/report.log 2>&1
log "QUEUE_DONE"
