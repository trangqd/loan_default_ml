from datetime import datetime     
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import entropy, ks_2samp

PRED_DIR = Path("data_mart/gold/predictions")
LABEL_ROOT = Path("data_mart/gold/label_store")
MON_DIR = Path("data_mart/gold/monitoring")
MON_DIR.mkdir(parents=True, exist_ok=True)
MON_FILE = MON_DIR / "monitoring_summary.parquet"


def _load_labels(snapshot_str: str) -> pd.DataFrame:
    """
    Load all parquet parts inside  label_store/snapshot_date=YYYY-MM-DD/
    snapshot_str comes in 'YYYY-MM'.
    """
    folder = LABEL_ROOT / f"snapshot_date={snapshot_str}-01"
    parts = sorted(folder.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)

def _merge_pred_label(pred_path: Path) -> pd.DataFrame | None:
    preds = pd.read_parquet(pred_path)
    if preds.empty:
        return None

    snapshot_str = pred_path.stem.replace("preds_", "")
    snapshot_dt = pd.to_datetime(f"{snapshot_str}-01")

    # Search across all label_store folders to find matching loan_ids
    all_parts = sorted(LABEL_ROOT.glob("snapshot_date=*/part-*.parquet"))
    if not all_parts:
        print(f"[!] No label files found at all.")
        return None

    label_dfs = []
    for part in all_parts:
        df = pd.read_parquet(part, columns=["loan_id", "label"])
        label_dfs.append(df)
    labels = pd.concat(label_dfs, ignore_index=True).drop_duplicates("loan_id")

    # Merge only on loan_id
    merged = preds.merge(labels, on="loan_id", how="inner")
    merged["snapshot_date"] = snapshot_dt

    return merged if not merged.empty else None


def compute_metrics(df: pd.DataFrame, snap: str, ref_df: pd.DataFrame | None):
    y_true = df["label"]
    y_hat = df["predicted_proba"]
    y_bin = (y_hat >= 0.5).astype(int)

    # ----- drift vs reference --------------------------------
    ks_target = ks_preds = np.nan
    if ref_df is not None:
        ks_target = round(ks_2samp(ref_df["label"], y_true).statistic, 4)
        ks_preds = round(ks_2samp(ref_df["predicted_proba"], y_hat).statistic, 4)

    # ----- outlier proportion in predictions -----------------
    outlier_pct = ((y_hat < 0.01) | (y_hat > 0.99)).mean()

    return {
        "snapshot_date": snap,
        "target_mean": round(y_true.mean(), 4),
        "pred_mean": round(y_hat.mean(), 4),
        "pred_std": round(y_hat.std(), 4),
        "pred_entropy": round(
            entropy(np.histogram(y_hat, bins=10, range=(0, 1))[0] + 1e-6), 4
        ),
        "auc": round(roc_auc_score(y_true, y_hat), 4),
        "gini": round(2 * roc_auc_score(y_true, y_hat) - 1, 4),
        "precision": round(precision_score(y_true, y_bin, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_bin, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_bin, zero_division=0), 4),
        "pct_outlier_preds": round(outlier_pct, 4),
        "ks_target": ks_target,
        "ks_preds": ks_preds,
    }

# ------------------------------------------------------------
def run_monitoring():
    pred_files = sorted(PRED_DIR.glob("preds_*.parquet"))
    if not pred_files:
        print("❌ No prediction files found.")
        return

    summary = []
    ref_df = None  # first month becomes reference

    for pf in pred_files:
        snap = pf.stem.replace("preds_", "")
        merged = _merge_pred_label(pf)
        if merged is None or merged.empty:
            print(f"[!] Skipped {snap}: empty merge")
            continue

        metrics = compute_metrics(merged, snap, ref_df)
        summary.append(metrics)

        if ref_df is None:
            ref_df = merged[["label", "predicted_proba"]]  # set reference

        print(f"[✓] Processed snapshot {snap}")
    
    if summary:
        df_summary = pd.DataFrame(summary)
        df_summary["snapshot_date"] = pd.to_datetime(df_summary["snapshot_date"] + "-01")
        df_summary.sort_values("snapshot_date", inplace=True)

        # 👉 give each file a unique name: monitoring_YYYYMMDD_HHMMSS.parquet
        run_ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_file = MON_DIR / f"monitoring_{run_ts}.parquet"

        df_summary.to_parquet(out_file, index=False)
        print(f"[✓] Monitoring summary saved to {out_file}")
    else:
        print("⚠️  No monitoring records produced.")

# ------------------------------------------------------------
if __name__ == "__main__":
    run_monitoring()
