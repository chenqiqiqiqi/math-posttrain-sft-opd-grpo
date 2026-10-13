"""构建项目 B 的全部数据集（协议第 2 节）。

产出（data/prepared/）：
  test_{aime24,aime25,amc23,aime26}.parquet  封存测试集
  sanity_math500.parquet                      评测器核验集
  dev_deepmath400.parquet                     开发集
  train_dapo.parquet                          D 族训练题（seed 42 固定题序）
  train_ot3.parquet                           Q 族训练题（seed 42 固定题序）
以及 manifests/data_manifest.json 与 reports/decontam/ 下的命中明细。

去污染：精确匹配（空白归一）+ all-mpnet-base-v2 余弦相似度 >= 0.6（论文附录 C.3 方法）。
所有文件统一为 verl 格式，提示模板统一为老师 RL 模板。
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream" / "Rethinking-OPD"
OUT = ROOT / "data" / "prepared"
REPORT = ROOT / "reports" / "decontam"
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
THRESHOLD = 0.6
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIR = ROOT / "data" / "raw" / "all-mpnet-base-v2"  # curl 下载的本地副本，revision 见目录内 REVISION
SEED = 42
DEV_SIZE = 400


def strip_suffix(text):
    # 上游 dapo-math-17k-processed 的模板误写成 oxed{{}}，两种都去掉
    text = text.strip()
    for suffix in (SUFFIX.strip(), SUFFIX.strip().replace("{}", "{{}}")):
        if text.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


def strip_boxed(ans):
    ans = str(ans).strip()
    m = re.fullmatch(r"\\boxed\{(.*)\}", ans, flags=re.S)
    return m.group(1).strip() if m else ans


def norm(q):
    return re.sub(r"\s+", " ", q).strip().lower()


def records(df, source, question_fn, answer_fn, id_fn):
    return pd.DataFrame(
        {
            "id": [id_fn(i, r) for i, r in df.iterrows()],
            "source": source,
            "question": [question_fn(r) for _, r in df.iterrows()],
            "answer": [strip_boxed(answer_fn(r)) for _, r in df.iterrows()],
        }
    )


def to_verl(rec):
    return pd.DataFrame(
        {
            "data_source": rec["source"],
            "prompt": [[{"role": "user", "content": q + SUFFIX}] for q in rec["question"]],
            "ability": "math",
            "reward_model": [{"ground_truth": a, "style": "rule"} for a in rec["answer"]],
            "extra_info": [
                {"index": i, "id": pid, "question": q}
                for i, (pid, q) in enumerate(zip(rec["id"], rec["question"]))
            ],
        }
    )


def load_sources():
    val = UP / "scripts" / "val" / "data"
    ds = UP / "datasets"
    prompt_q = lambda r: strip_suffix(r["prompt"][0]["content"])
    gt = lambda r: r["reward_model"]["ground_truth"]
    src = {}
    for name in ["AIME24", "AIME25", "AMC23"]:
        df = pd.read_parquet(val / name / "test.parquet")
        src[f"test_{name.lower()}"] = records(df, name, prompt_q, gt, lambda i, r, n=name: f"{n}-{i}")
    df = pd.read_parquet(ROOT / "data" / "raw" / "aime_2026.parquet")
    src["test_aime26"] = records(
        df, "AIME26", lambda r: r["problem"].strip(), lambda r: r["answer"], lambda i, r: f"AIME26-{r['problem_idx']}"
    )
    df = pd.read_parquet(val / "MATH-500" / "test.parquet")
    src["sanity_math500"] = records(df, "MATH-500", prompt_q, gt, lambda i, r: f"MATH500-{i}")
    df = pd.read_parquet(ds / "dapo-math-17k-processed.parquet")
    src["train_dapo"] = records(df, "DAPO-Math-17k", prompt_q, gt, lambda i, r: f"DAPO-{r['extra_info']['index']}")
    df = pd.read_parquet(ds / "OpenThoughts3_opd.parquet")
    src["train_ot3"] = records(df, "OpenThoughts3-opd", prompt_q, gt, lambda i, r: f"OT3-{r['extra_info']['index']}")
    df = pd.read_parquet(ds / "DeepMath_deduped.parquet")
    src["dev_pool"] = records(
        df, "DeepMath-103K", lambda r: str(r["extra_info"]["question"]).strip(), gt,
        lambda i, r: f"DeepMath-{r['extra_info']['index']}",
    )
    return src


def find_hits(model, cand, ref, cache):
    """cand 中每题对 ref 的精确命中与 top-1 余弦相似度。"""
    ref_set = set(norm(q) for q in ref["question"])
    exact = np.array([norm(q) in ref_set for q in cand["question"]])

    def emb(df):
        key = tuple(df["id"])
        if key not in cache:
            cache[key] = model.encode(
                df["question"].tolist(), batch_size=128, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=True,
            )
        return cache[key]

    sims = emb(cand) @ emb(ref).T
    best, idx = sims.max(dim=1)
    return exact, best.float().cpu().numpy(), idx.cpu().numpy()


def decontam(model, name, cand, refs, cache):
    """去掉与 refs 中任一集合命中的题，返回保留部分并写命中明细。"""
    remove = np.zeros(len(cand), dtype=bool)
    stats = {}
    rows = []
    for ref_name, ref in refs.items():
        exact, best, idx = find_hits(model, cand, ref, cache)
        hit = exact | (best >= THRESHOLD)
        stats[ref_name] = {"exact": int(exact.sum()), "semantic_only": int((hit & ~exact).sum()), "total": int(hit.sum())}
        for i in np.where(hit)[0]:
            rows.append(
                {
                    "cand_id": cand["id"].iloc[i], "ref_set": ref_name, "ref_id": ref["id"].iloc[idx[i]],
                    "exact": bool(exact[i]), "cosine": round(float(best[i]), 4),
                    "cand_question": cand["question"].iloc[i][:300], "ref_question": ref["question"].iloc[idx[i]][:300],
                }
            )
        remove |= hit
    if rows:
        pd.DataFrame(rows).sort_values("cosine", ascending=False).to_csv(
            REPORT / f"{name}_hits.csv", index=False, encoding="utf-8-sig"
        )
    stats["removed"] = int(remove.sum())
    stats["kept"] = int((~remove).sum())
    return cand[~remove].reset_index(drop=True), stats


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    src = load_sources()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(EMBED_DIR), device=device, local_files_only=True)
    cache = {}
    manifest = {"seed": SEED, "threshold": THRESHOLD, "embed_model": EMBED_MODEL, "embed_revision": (EMBED_DIR / "REVISION").read_text().strip(), "sets": {}, "decontam": {}}

    tests = {k: v for k, v in src.items() if k.startswith("test_")}
    test_all = pd.concat(tests.values(), ignore_index=True)
    for k, v in tests.items():
        dup = v["question"].map(norm).duplicated().sum()
        assert dup == 0, f"{k} 内有重复题"

    # 训练集只对测试集去污染
    train = {}
    for name in ["train_dapo", "train_ot3"]:
        train[name], manifest["decontam"][name] = decontam(model, name, src[name], {"test": test_all}, cache)

    # 开发集候选：对测试集与两个训练集都去污染（DeepMath 已由论文对 DAPO 去重，这里复核）
    dev_pool, manifest["decontam"]["dev_pool"] = decontam(
        model, "dev_pool", src["dev_pool"],
        {"test": test_all, "train_dapo": train["train_dapo"], "train_ot3": train["train_ot3"]}, cache,
    )
    rng = np.random.default_rng(SEED)
    dev = dev_pool.iloc[np.sort(rng.choice(len(dev_pool), DEV_SIZE, replace=False))].reset_index(drop=True)

    # 固定题序：seed 42 打乱后训练时 data.shuffle=False
    for name in train:
        perm = np.random.default_rng(SEED).permutation(len(train[name]))
        train[name] = train[name].iloc[perm].reset_index(drop=True)

    outputs = {**tests, "sanity_math500": src["sanity_math500"], "dev_deepmath400": dev, **train}
    for name, rec in outputs.items():
        path = OUT / f"{name}.parquet"
        to_verl(rec).to_parquet(path, index=False)
        manifest["sets"][name] = {"rows": len(rec), "sha256": sha256(path)}
        print(f"{name}: {len(rec)}")
    (ROOT / "manifests" / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["decontam"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
