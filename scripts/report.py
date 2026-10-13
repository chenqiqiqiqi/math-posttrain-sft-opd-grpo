"""结果汇总（协议第 4 节）：读 runs/eval/test/<模型>/graded.csv，输出 Markdown 表与 JSON。

- 主口径 clean（去掉老师训练数据重叠的 29 题，共 101 题）；另报 all（130 题）与 4 个子集。
- 每个模型：pass@k（k ≤ 采样数）、全错题数、全对题数、截断率、平均长度。
- 配对比较（bootstrap 95% CI + 符号检验）：各训练组 vs 基线；D-OPD vs D-GRPO；D-OPD vs D-SFT；
  Q 族：Q-SFT-OPD vs Q-OPD（复现判据 b）、Q-SFT-OPD vs Q-SFT。
- 能力边界：基线 0/n 的题中，各模型至少做对 1 次的题数。

用法：python scripts/report.py [--eval-dir runs/eval/test] [--out reports/results]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stats import boundary, paired, select, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ["clean", "all", "test_aime24", "test_aime25", "test_aime26", "test_amc23"]
PAIRS = [("D-SFT", "D0"), ("D-OPD", "D0"), ("D-GRPO", "D0"), ("D-OPD", "D-GRPO"), ("D-OPD", "D-SFT"),
         ("D-SFT", "D-GRPO"), ("D-T", "D0"),
         ("Q-OPD", "Q-Base"), ("Q-SFT-OPD", "Q-OPD"), ("Q-SFT-OPD", "Q-SFT"), ("Q-SFT", "Q-Base")]


def fmt(x, pct=True):
    return "—" if x is None else (f"{100 * x:.1f}" if pct else f"{x:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval" / "test"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {p.parent.name: pd.read_csv(p) for p in sorted(Path(args.eval_dir).glob("*/graded.csv"))}
    result = {"models": {}, "pairs": {}, "boundary": {}}
    md = ["# 测试集结果（主口径 clean = 101 题）", ""]

    for sub in SUBSETS:
        md += [f"## 子集：{sub}", "", "| 模型 | 题数 | n | pass@1 | pass@4 | pass@16 | pass@32 | 全错 | 全对 | 截断率 | 平均长度 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, t in tables.items():
            s = summarize(select(t, sub))
            result["models"].setdefault(name, {})[sub] = s
            md.append(f"| {name} | {s['problems']} | {int(t['n'].min())} | {fmt(s.get('pass@1'))} | {fmt(s.get('pass@4'))} | "
                      f"{fmt(s.get('pass@16'))} | {fmt(s.get('pass@32'))} | {s['solve_none']} | {s['solve_all']} | "
                      f"{fmt(s['truncation_rate'])} | {s['mean_tokens']:.0f} |")
        md.append("")

    md += ["## 配对比较（主口径 clean，A − B，百分点）", "",
           "| A | B | k | 差值 | 95% CI | 胜/负 | 符号检验 p |", "|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        if a in tables and b in tables:
            ta, tb = select(tables[a], "clean"), select(tables[b], "clean")
            nmin = int(min(ta["n"].min(), tb["n"].min()))
            for k in [1, nmin]:
                r = paired(ta, tb, k)
                result["pairs"][f"{a}-{b}@{k}"] = r
                md.append(f"| {a} | {b} | {k} | {100 * r['diff']:+.1f} | [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}] | "
                          f"{r['wins']}/{r['losses']} | {r['sign_p']:.3f} |")
    md += ["", "## 能力边界（主口径 clean）：基线全错的题中，至少做对 1 次的题数", ""]
    for name, base in [(n, "D0") for n in ["D-SFT", "D-OPD", "D-GRPO", "D-T"]] + [(n, "Q-Base") for n in ["Q-SFT", "Q-OPD", "Q-SFT-OPD"]]:
        if name in tables and base in tables:
            r = boundary(select(tables[name], "clean"), select(tables[base], "clean"))
            result["boundary"][f"{name}|{base}"] = r
            md.append(f"- {name}（相对 {base}）：{r['newly_solved']} / {r['base_unsolved']}")
    (out / "test_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
