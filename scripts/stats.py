"""统计：pass@k 无偏估计、逐题配对 bootstrap CI、符号检验、能力边界计数（协议第 4 节）。

输入是 eval_grade.py 生成的 graded.csv（每题 n 次采样中正确 c 次）。
示例：python scripts/stats.py runs/eval/test/D-OPD/graded.csv runs/eval/test/S0/graded.csv --subset clean
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 2, 4, 8, 16, 32, 64)


def pass_at_k(n, c, k):
    """Chen et al. 2021 无偏估计：1 - C(n-c, k) / C(n, k)。"""
    if k > n:
        return np.nan
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def per_problem(table, k):
    return np.array([pass_at_k(n, c, k) for n, c in zip(table["n"], table["correct"])])


def select(table, subset):
    """subset: all / clean（去掉老师训练数据重叠题）/ 某个集合名（如 test_aime24）。"""
    if subset == "all":
        return table
    if subset == "clean":
        overlap = set(json.loads((ROOT / "manifests" / "test_teacher_overlap.json").read_text(encoding="utf-8"))["overlap_ids"])
        return table[~table["id"].isin(overlap)]
    return table[table["set"] == subset]


def summarize(table, ks=KS):
    out = {"problems": len(table)}
    for k in ks:
        if k <= table["n"].min():
            out[f"pass@{k}"] = float(per_problem(table, k).mean())
    out["solve_none"] = int((table["correct"] == 0).sum())
    out["solve_all"] = int((table["correct"] == table["n"]).sum())
    out["truncation_rate"] = float(table["truncated"].sum() / table["n"].sum())
    out["mean_tokens"] = float((table["mean_tokens"] * table["n"]).sum() / table["n"].sum())
    return out


def paired(a, b, k, n_boot=10_000, seed=0):
    """a − b 的逐题配对比较：均值差、bootstrap 95% CI、符号检验 p 值。"""
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    pa = np.array([pass_at_k(n, c, k) for n, c in zip(m["n_a"], m["correct_a"])])
    pb = np.array([pass_at_k(n, c, k) for n, c in zip(m["n_b"], m["correct_b"])])
    d = pa - pb
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    p_sign = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
    return {
        "k": k, "problems": len(d), "diff": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "wins": pos, "losses": neg, "sign_p": float(p_sign),
    }


def boundary(model, base):
    """base 一次都没做对（0/n）的题中，model 至少做对一次的题数。"""
    m = model.merge(base, on="id", suffixes=("", "_base"))
    zero = m[m["correct_base"] == 0]
    return {"base_unsolved": len(zero), "newly_solved": int((zero["correct"] > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="模型 A 的 graded.csv")
    ap.add_argument("b", help="模型 B（对照）的 graded.csv")
    ap.add_argument("--subset", default="clean")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 64])
    args = ap.parse_args()
    a, b = select(pd.read_csv(args.a), args.subset), select(pd.read_csv(args.b), args.subset)
    print(json.dumps({
        "subset": args.subset, "A": summarize(a), "B": summarize(b),
        "A_minus_B": [paired(a, b, k) for k in args.ks], "boundary_A_vs_B": boundary(a, b),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
