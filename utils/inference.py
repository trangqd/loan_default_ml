import argparse
import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.impute import SimpleImputer

# --- internal project modules -------------------------------------------
from utils import data_io
from utils.modelling import DROP_COLS  # use same columns list as training

MODEL_DIR = "model_bank"

def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_best_model(metric="auc_oot") -> dict:
    best_art  = None
    best_val  = -float("inf")
    best_file = None

    for fn in os.listdir(MODEL_DIR):
        if not fn.endswith(".pkl"):
            continue
        art = _load_pkl(os.path.join(MODEL_DIR, fn))
        results = art.get("results", {})
        val = results.get(metric) or results.get("auc_test")
        if val is not None and val > best_val:
            best_val  = val
            best_art  = art
            best_file = fn

    if best_art is None:
        raise FileNotFoundError("No model artefacts with metric found in model_bank/")

    print(f"[\u2713] Selected best model '{best_file}'  ({metric}={best_val:.4f})")
    return best_art

def preprocess_features_inference(X_raw, artefact):
    base_drop = list(DROP_COLS) + ["snapshot_date"]
    X = X_raw.drop(columns=base_drop, errors="ignore").copy()

    # Business-rule fills
    for col, default in [("Credit_Mix_Encoded", 1),
                         ("is_min_pay_only", 1),
                         ("Payment_Behaviour_Encoded", 0)]:
        if col in X.columns:
            X[col] = X[col].fillna(default)

    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Use imputer from artefact
    imputer = artefact["preprocessing_transformers"].get("imputer")
    if imputer is None:
        raise ValueError("Missing trained imputer in artefact.")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure all expected numeric features exist (fill missing ones with NaN)
    expected = list(imputer.feature_names_in_)
    for col in expected:
        if col not in X.columns:
            X[col] = np.nan
    X = X[expected]

    arr = imputer.transform(X)
    out_cols = imputer.get_feature_names_out(expected)
    df_imp = pd.DataFrame(arr, columns=out_cols, index=X.index)

    return df_imp


def run_inference(snapshot_date_str: str):
    snapshot_dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    artefact = load_best_model()
    model  = artefact["model"]
    scaler = artefact["preprocessing_transformers"]["scaler"]

    month_start = snapshot_dt.replace(day=1)
    month_end   = month_start + relativedelta(months=1) - timedelta(days=1)

    feat = data_io.load_feature_store(
        date_from=month_start.strftime("%Y-%m-%d"),
        date_to=month_end.strftime("%Y-%m-%d"),
        filter_leakage=True
    )

    if feat.empty:
        print(f"[!] Skipped {month_start:%Y-%m} – No feature data.")
        return

    feat.set_index("loan_id", inplace=True)
    X_inf   = preprocess_features_inference(feat, artefact)
    X_scaled = scaler.transform(X_inf)
    probs = model.predict_proba(X_scaled)[:, 1]

    df_preds = (
        pd.DataFrame({
            "loan_id": X_inf.index,
            "snapshot_date": snapshot_date_str,
            "predicted_proba": probs
        })
        .reset_index(drop=True)
    )

    data_io.save_predictions(df_preds)
    print(f"[\u2713] Inference complete – {df_preds.shape[0]:,} rows scored.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference for a snapshot month")
    parser.add_argument("--snapshotdate", required=True, type=str,
                        help="Snapshot date YYYY-MM-DD (must be first of month)")
    args = parser.parse_args()

    run_inference(args.snapshotdate)
