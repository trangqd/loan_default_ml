import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

# ---- Global Paths -------------------------------------------------------
DATA_MART_ROOT = Path(__file__).resolve().parent.parent / "data_mart"
print("DATA_MART_ROOT →", DATA_MART_ROOT.resolve())

GOLD_DIR         = DATA_MART_ROOT / "gold"
GOLD_FEATURE_DIR = GOLD_DIR / "feature_store" / "full"
GOLD_LABEL_DIR   = GOLD_DIR / "label_store"
GOLD_PRED_DIR    = GOLD_DIR / "predictions"
GOLD_MON_DIR     = GOLD_DIR / "monitoring"

for _d in [GOLD_PRED_DIR, GOLD_MON_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---- Internal Helpers ---------------------------------------------------
def _list_parquet(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())

def _concat_parquet(parts: List[Path]) -> pd.DataFrame:
    if not parts:
        raise FileNotFoundError(f"No parquet found under {parts}")
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)

def _filter_by_date(
    df: pd.DataFrame,
    date_from: Optional[str],
    date_to: Optional[str],
    col: str = "snapshot_date"
) -> pd.DataFrame:
    if date_from or date_to:
        df[col] = pd.to_datetime(df[col])
        if date_from:
            df = df[df[col] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df[col] <= pd.to_datetime(date_to)]
    return df

def _partition_and_save(df: pd.DataFrame, root: Path, prefix: str) -> None:
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    for period, grp in df.groupby(df["snapshot_date"].dt.to_period("M")):
        out = root / f"{prefix}_{period}.parquet"
        grp.to_parquet(out, index=False)
        print(f"[✓] Saved {out}  ({grp.shape[0]:,} rows)")

# ---- Loaders ------------------------------------------------------------
def load_feature_store(date_from=None, date_to=None, filter_leakage=True):
    parts = _list_parquet(GOLD_FEATURE_DIR)
    df = _concat_parquet(parts)
    df = _filter_by_date(df, date_from, date_to)

    if filter_leakage:
        df["loan_start_date"] = pd.to_datetime(df["loan_start_date"], errors="coerce")
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        df = df[df["snapshot_date"] == df["loan_start_date"]]

    return df

def load_label_store(date_from=None, date_to=None, label_col="label"):
    parts = _list_parquet(GOLD_LABEL_DIR)
    df = _concat_parquet(parts)
    df = _filter_by_date(df, date_from, date_to)
    if label_col != "label":
        df = df.rename(columns={"label": label_col})
    return df

def load_dataset(date_from=None, date_to=None, label_col="label", filter_leakage=True):
    feat = load_feature_store(date_from, date_to, filter_leakage=filter_leakage)
    lab  = load_label_store(label_col=label_col)
    df = feat.merge(lab[["loan_id", label_col]], on="loan_id", how="inner", validate="m:1")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


# ---- Writers ------------------------------------------------------------
def save_predictions(df_preds: pd.DataFrame):
    _partition_and_save(df_preds, GOLD_PRED_DIR, prefix="preds")

def save_monitoring(df_monitor: pd.DataFrame):
    _partition_and_save(df_monitor, GOLD_MON_DIR, prefix="monitor")