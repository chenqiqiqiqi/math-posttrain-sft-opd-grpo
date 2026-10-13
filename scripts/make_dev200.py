"""补充 1：开发集缩为 200 题——从 dev_deepmath400 中用 seed 42 随机抽 200 题（不看任何成绩）。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "data" / "prepared" / "dev_deepmath400.parquet"
out = ROOT / "data" / "prepared" / "dev_deepmath200.parquet"
df = pd.read_parquet(src)
idx = np.sort(np.random.default_rng(42).choice(len(df), 200, replace=False))
df.iloc[idx].reset_index(drop=True).to_parquet(out, index=False)
m_path = ROOT / "manifests" / "data_manifest.json"
m = json.loads(m_path.read_text(encoding="utf-8"))
m["sets"]["dev_deepmath200"] = {"rows": 200, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                "derived_from": "dev_deepmath400, seed 42"}
m_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
print(m["sets"]["dev_deepmath200"])
