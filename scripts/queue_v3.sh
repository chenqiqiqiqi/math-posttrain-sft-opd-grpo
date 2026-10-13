#!/bin/bash
# 排队 v3（协议补充 8）：接续被停止的 Q-OPD（第 40 步），Q-SFT-OPD 改为 40 步；其余与 queue.sh 相同。
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

# 用法：setsid nohup bash scripts/queue_v3.sh <Q-OPD 的 run_arm.sh 进程号> > runs/queue_v3.log 2>&1 &
log "QUEUE_START (v3, 补充 8：Q 族各 40 步) waiting for Q-OPD run_arm pid=$1 to exit"
while kill -0 "$1" 2>/dev/null; do sleep 10; done
cleanup_gpu
if [ -f runs/train/Q-OPD/hf/step_40/model.safetensors ]; then
  echo "STOPPED_AMEND8 final_step=40 $(date -Is)（协议补充 8：Q 族改为 40 步；终点 hf/step_40 已导出）" >> runs/train/Q-OPD/arm_summary.txt
  log "DONE Q-OPD (补充 8, final_step=40)"
else
  log "QUEUE_STOPPED Q-OPD step_40 导出缺失（需人工处理）"; exit 1
fi
dev_and_prune Q-OPD
run_with_retry Q-SFT-OPD 66 env ARM=Q-OPD STUDENT=$M/QS TEACHER=$M/QT RUN_NAME=Q-SFT-OPD MAX_STEPS=40 SAVE_FREQ=20 PB_BUDGET_SEC=0 bash scripts/run_arm.sh
dev_and_prune Q-SFT-OPD
log "TRAINING_DONE all arms"

# 基线开发集评测；测试集评测（所有模型已冻结，只此一次）；汇总
log "DEV_EVAL baselines"
python scripts/eval_jobs.py --plan dev > runs/eval_dev_rest.log 2>&1; cleanup_gpu
log "TEST_EVAL start (all models frozen)"
python scripts/eval_jobs.py --plan test > runs/eval_test.log 2>&1; cleanup_gpu
log "TEST_EVAL done"
python scripts/report.py > runs/report.log 2>&1
log "QUEUE_DONE"
