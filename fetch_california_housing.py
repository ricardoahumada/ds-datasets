from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

OUT_DIR = Path("M10/2026/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = fetch_california_housing(as_frame=True)
df: pd.DataFrame = data.frame
df["target"] = data.target

out_path = OUT_DIR / "california_housing.parquet"
df.to_parquet(out_path, engine="pyarrow", index=False)

print(f"rows={len(df):,} cols={df.shape[1]}")
print(f"path={out_path.resolve()}")
print(f"size={out_path.stat().st_size / 1024:.1f} KB")