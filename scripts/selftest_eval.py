"""评测流程的本地自测（不需要 GPU、不涉及真实模型）：
1) pass@k 无偏估计与穷举一致；2) 判分器识别常见格式；3) 合成两个"模型"走完判分 + 统计全流程。
"""
import itertools
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import eval_grade  # noqa: E402
import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_pass_at_k():
    for n, c, k in [(8, 3, 2), (8, 0, 4), (8, 8, 1), (6, 2, 3), (10, 1, 5)]:
        combos = list(itertools.combinations(range(n), k))
        brute = np.mean([any(i < c for i in s) for s in combos])
        assert abs(brute - stats.pass_at_k(n, c, k)) < 1e-12, (n, c, k)


def test_grader():
    g = eval_grade.grade_answer_verl
    cases = [
        (r"<think>...</think> so \boxed{204}", "204", True),
        (r"\boxed{204}", r"\boxed{204}", True),
        (r"answer \boxed{\frac{1}{2}}", r"\frac{1}{2}", True),
        (r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}", True),
        (r"\boxed{205}", "204", False),
        ("204", "204", False),  # 没有 \boxed 不算
        (r"\boxed{1} ... final \boxed{204}", "204", True),  # 取最后一个
    ]
    for resp, gt, want in cases:
        assert bool(g(resp, gt)) == want, (resp, gt)


def test_pipeline():
    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp())
    for name, p in [("A", 0.6), ("B", 0.3)]:
        d = tmp / name
        d.mkdir()
        for f in ["test_aime24", "test_amc23", "test_aime26"]:
            df = pd.read_parquet(ROOT / "data" / "prepared" / f"{f}.parquet")
            with (d / f"{f}.jsonl").open("w", encoding="utf-8") as fh:
                for _, r in df.iterrows():
                    gt = r["reward_model"]["ground_truth"]
                    for j in range(8):
                        ans = gt if rng.random() < p else "99999"
                        fh.write(json.dumps({
                            "id": r["extra_info"]["id"], "set": f, "sample_idx": j,
                            "response": "<think>x</think> the answer is \\boxed{" + ans + "}",
                            "num_tokens": 100, "finish_reason": "length" if j == 0 else "stop",
                        }) + "\n")
    ta, tb = eval_grade.grade_dir(tmp / "A"), eval_grade.grade_dir(tmp / "B")
    acc_a, acc_b = ta["correct"].sum() / ta["n"].sum(), tb["correct"].sum() / tb["n"].sum()
    assert 0.5 < acc_a < 0.7 and 0.2 < acc_b < 0.4, (acc_a, acc_b)
    a, b = stats.select(ta, "clean"), stats.select(tb, "clean")
    assert len(a) == 30 + 29 + 12, len(a)  # AIME24 30 + AIME26 去 1 + AMC23 去 28
    r = stats.paired(a, b, 1)
    assert r["ci95"][0] > 0, r
    assert abs(stats.summarize(a)["truncation_rate"] - 1 / 8) < 1e-9
    print(f"合成 A={acc_a:.3f} B={acc_b:.3f}；A−B pass@1={r['diff']:.3f} CI={np.round(r['ci95'], 3).tolist()}")


if __name__ == "__main__":
    test_pass_at_k()
    test_grader()
    test_pipeline()
    print("全部自测通过")
