"""对 eval_generate.py 的输出判分，生成每题正确次数表（协议第 4 节）。

判分 = 论文独立评测流程的规则判分 grade_answer_verl（mathd + sympy，取最后一个 \\boxed{}），不开 LLM 判分器。
输出 <dir>/graded.csv：id, set, n, correct, truncated, mean_tokens

示例：python scripts/eval_grade.py runs/eval/test/S0
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "Rethinking-OPD" / "scripts" / "val" / "eval"))
from utils import grade_answer_verl  # noqa: E402


def load_answers():
    ans = {}
    for p in (ROOT / "data" / "prepared").glob("*.parquet"):
        if p.stem.startswith("train_"):
            continue
        df = pd.read_parquet(p, columns=["reward_model", "extra_info"])
        for rm, ex in zip(df["reward_model"], df["extra_info"]):
            ans[ex["id"]] = rm["ground_truth"]
    return ans


def grade_one(args):
    response, gt = args
    try:
        return bool(grade_answer_verl(response, gt))
    except Exception:
        return False


def grade_dir(run_dir, workers=8):
    run_dir = Path(run_dir)
    answers = load_answers()
    rows = []
    for f in sorted(run_dir.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    df = pd.DataFrame(rows)
    with ProcessPoolExecutor(workers) as ex:
        df["ok"] = list(ex.map(grade_one, [(r, answers[i]) for r, i in zip(df["response"], df["id"])], chunksize=64))
    df["truncated"] = df["finish_reason"] == "length"
    table = df.groupby(["id", "set"], sort=False).agg(
        n=("ok", "size"), correct=("ok", "sum"), truncated=("truncated", "sum"), mean_tokens=("num_tokens", "mean")
    ).reset_index()
    table.to_csv(run_dir / "graded.csv", index=False)
    return table


if __name__ == "__main__":
    t = grade_dir(sys.argv[1])
    print(t.groupby("set").agg(problems=("id", "size"), acc=("correct", "sum"), n=("n", "sum")).assign(
        acc=lambda x: (x.acc / x.n).round(4)))
