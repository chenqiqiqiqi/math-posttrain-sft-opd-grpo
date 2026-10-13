"""评测排队（协议第 4 节 + 补充 1）：两张卡各开一个工作进程，从任务表里取任务，已完成的跳过（可断点续跑）。

任务表由 --plan 指定：
  dev   开发集 200 题 × k=8、最长 16,384：D0、D-T、各训练组的 bf16 导出（hf/step_*）、Q 族基线与导出
  test  封存测试集 130 题、最长 31,744：D 族 k=32（D0、D-T、三个训练组终点），Q 族 k=16（基线与两组终点）
  Q 族开发集与测试集最长生成均为 8,192（协议补充 9）
        —— 只在所有模型冻结后运行一次（协议第 2 节）
每个任务：eval_generate.py 生成 → eval_grade.py 判分，输出在 runs/eval/<plan>/<name>/。

用法：python scripts/eval_jobs.py --plan dev [--only D-OPD] [--dry-run]
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Process
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = Path("/root/autodl-tmp/models")
TRAIN = ROOT / "runs" / "train"
DATA = ROOT / "data" / "prepared"
TEST_SETS = [DATA / f"test_{s}.parquet" for s in ["aime24", "aime25", "amc23", "aime26"]]
D_ARMS = ["D-SFT", "D-OPD", "D-GRPO"]
Q_MAX_TOKENS = 8192  # 协议补充 9：Q 族（基线与两组）开发集与测试集最长生成均为 8,192
Q_ARMS = ["Q-OPD", "Q-SFT-OPD"]


def exports(arm):
    d = TRAIN / arm / "hf"
    return sorted((int(p.name.split("_")[1]), p) for p in d.glob("step_*") if (p / "model.safetensors").exists())


def plan_jobs(plan):
    jobs = []  # (name, model_path, data_files, k, max_tokens, enable_thinking)
    if plan == "dev":
        dev = [DATA / "dev_deepmath200.parquet"]
        jobs += [("D0", M / "S0", dev, 8, 16384, "none"), ("D-T", M / "T", dev, 8, 16384, "none"),
                 ("Q-Base", M / "QB", dev, 8, Q_MAX_TOKENS, "false"), ("Q-SFT", M / "QS", dev, 8, Q_MAX_TOKENS, "false")]
        for arm in D_ARMS + Q_ARMS:
            think = "false" if arm.startswith("Q") else "none"
            max_tokens = Q_MAX_TOKENS if arm.startswith("Q") else 16384
            jobs += [(f"{arm}@{s}", p, dev, 8, max_tokens, think) for s, p in exports(arm)]
    elif plan == "test":
        jobs += [("D0", M / "S0", TEST_SETS, 32, 31744, "none"), ("D-T", M / "T", TEST_SETS, 32, 31744, "none")]
        for arm in D_ARMS:
            ex = exports(arm)
            if ex:
                jobs.append((arm, ex[-1][1], TEST_SETS, 32, 31744, "none"))  # 终点 = 最大步数
        jobs += [("Q-Base", M / "QB", TEST_SETS, 16, Q_MAX_TOKENS, "false"), ("Q-SFT", M / "QS", TEST_SETS, 16, Q_MAX_TOKENS, "false")]
        for arm in Q_ARMS:
            ex = exports(arm)
            if ex:
                jobs.append((arm, ex[-1][1], TEST_SETS, 16, Q_MAX_TOKENS, "false"))
    return jobs


def done(plan, name, data_files):
    out = ROOT / "runs" / "eval" / plan / name
    return (out / "graded.csv").exists() and all((out / f"{Path(f).stem}.jsonl").exists() for f in data_files)


def run_job(plan, job, gpu):
    name, model, data_files, k, max_tokens, think = job
    out = ROOT / "runs" / "eval" / plan / name
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with open(out / "eval.log", "a") as log:
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_generate.py"), "--model", str(model),
                             "--name", name, "--data", *map(str, data_files), "--k", str(k),
                             "--max-tokens", str(max_tokens), "--enable-thinking", think, "--out", str(out)],
                            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_grade.py"), str(out)],
                                env={**env, "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=subprocess.STDOUT).returncode
    status = "OK" if rc == 0 else f"FAILED rc={rc}"
    with open(ROOT / "runs" / "eval" / f"{plan}_status.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {name} gpu{gpu} {status}\n")


def worker(plan, jobs, gpu):
    for job in jobs:
        if not done(plan, job[0], job[2]):
            run_job(plan, job, gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["dev", "test"], required=True)
    ap.add_argument("--only", nargs="*", help="只跑名字以这些前缀开头的任务")
    ap.add_argument("--exact", action="store_true", help="--only 按完整名字匹配（Q-SFT 不再匹配 Q-SFT-OPD）")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = plan_jobs(args.plan)
    if args.only:
        jobs = [j for j in jobs if (j[0] in args.only if args.exact else any(j[0].startswith(o) for o in args.only))]
    jobs = [j for j in jobs if not done(args.plan, j[0], j[2])]
    gpus = args.gpus.split(",")
    shards = [jobs[i::len(gpus)] for i in range(len(gpus))]  # 轮流分配
    for g, sh in zip(gpus, shards):
        print(f"gpu{g}:", [j[0] for j in sh])
    if args.dry_run:
        return
    (ROOT / "runs" / "eval").mkdir(parents=True, exist_ok=True)
    procs = [Process(target=worker, args=(args.plan, sh, g)) for g, sh in zip(gpus, shards)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
