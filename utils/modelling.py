import os
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb


# ───────────────────────────────────────────────────────────────────────────
# 1. Columns never allowed into the model
# ───────────────────────────────────────────────────────────────────────────
DROP_COLS = [
    "loan_id", "Customer_ID", "loan_start_date", "mob", "age_band_encoded",
    "dpd", "first_missed_date",
    "overdue_amt", "Delay_from_due_date",
    "payment_ratio", "shortfall", "full_payment", "overpayment",
    "missed_payment", "installments_missed",
    "rolling_avg_payment_ratio_3m", "rolling_sum_shortfall_3m",
    "rolling_max_dpd_3m", "consecutive_missed_payments",
]


# ───────────────────────────────────────────────────────────────────────────
# 2. Split into Train / Test / OOT using snapshot windows
# ───────────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame, cfg: Dict[str, str]) -> Tuple[pd.DataFrame, ...]:
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    # Convert config dates to Timestamp
    cfg = cfg.copy()
    cfg["train_test_start_date"] = pd.to_datetime(cfg["train_test_start_date"])
    cfg["train_test_end_date"]   = pd.to_datetime(cfg["train_test_end_date"])
    cfg["oot_start_date"]        = pd.to_datetime(cfg["oot_start_date"])
    cfg["oot_end_date"]          = pd.to_datetime(cfg["oot_end_date"])

    feature_cols = [c for c in df.columns if c not in ["label", "snapshot_date"]]

    # -- Train + Test window
    train_test = df[
        (df["snapshot_date"] >= cfg["train_test_start_date"]) &
        (df["snapshot_date"] <= cfg["train_test_end_date"])
    ]
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        train_test[feature_cols], train_test["label"],
        test_size=1 - cfg["train_test_ratio"],
        random_state=88, shuffle=True, stratify=train_test["label"]
    )

    # -- OOT window
    oot = df[
        (df["snapshot_date"] >= cfg["oot_start_date"]) &
        (df["snapshot_date"] <= cfg["oot_end_date"])
    ]
    if oot.empty:
        raise ValueError("OOT window contains 0 rows.")

    return (
        X_tr_raw.reset_index(drop=True),
        X_te_raw.reset_index(drop=True),
        oot[feature_cols].reset_index(drop=True),
        y_tr.reset_index(drop=True),
        y_te.reset_index(drop=True),
        oot["label"].reset_index(drop=True),
    )


# ───────────────────────────────────────────────────────────────────────────
# 3. Pre-processing helpers
# ───────────────────────────────────────────────────────────────────────────
def preprocess_features(
    X_raw: pd.DataFrame, imputer: SimpleImputer | None = None
) -> Tuple[pd.DataFrame, SimpleImputer]:
    """
    • Drops leakage columns.
    • Fills a couple of business-logic defaults.
    • Imputes numeric columns (median) – **fit only when imputer is None**.
    """
    X = X_raw.drop(columns=DROP_COLS, errors="ignore").copy()

    # business defaults
    defaults = {
        "Credit_Mix_Encoded":        1,
        "is_min_pay_only":           1,
        "Payment_Behaviour_Encoded": 0,
    }
    for col, val in defaults.items():
        if col in X:
            X[col] = X[col].fillna(val)

    # object → numeric
    for col in X.select_dtypes(include="object"):
        X[col] = pd.to_numeric(X[col], errors="coerce")

    num_cols = X.select_dtypes(include=[np.number]).columns
    if imputer is None:
        imputer = SimpleImputer(strategy="median", add_indicator=True).fit(X[num_cols])

    X_imp = pd.DataFrame(
        imputer.transform(X[num_cols]),
        columns=imputer.get_feature_names_out(num_cols),
        index=X.index,
    )
    return pd.concat([X.drop(columns=num_cols), X_imp], axis=1), imputer


def scale_features(
    X_tr: pd.DataFrame, X_te: pd.DataFrame, X_oot: pd.DataFrame
) -> Tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X_tr)
    return (
        scaler,
        scaler.transform(X_tr),
        scaler.transform(X_te),
        scaler.transform(X_oot),
    )


# ───────────────────────────────────────────────────────────────────────────
# 4. Model training + evaluation (unchanged)
# ───────────────────────────────────────────────────────────────────────────
def train_xgboost_with_search(
    X_tr, y_tr, n_iter: int = 30
) -> Tuple[xgb.XGBClassifier, Dict[str, Any], float]:
    base = xgb.XGBClassifier(eval_metric="logloss", random_state=88, nthread=1)

    param_dist = {
        "n_estimators":     [400, 600],
        "max_depth":        [6, 8],
        "learning_rate":    [0.01, 0.05],
        "subsample":        [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "gamma":            [0, 0.1],
        "min_child_weight": [1, 3, 5],
        "reg_alpha":        [0.1, 1],
        "reg_lambda":       [1, 1.5, 2],
    }

    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=3,
        scoring=make_scorer(roc_auc_score),
        n_jobs=1, random_state=42, verbose=1
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, proba)
    return auc, 2 * auc - 1   # AUC, Gini


# ───────────────────────────────────────────────────────────────────────────
# 5. Artefact builder + persistence (same API as before)
# ───────────────────────────────────────────────────────────────────────────
def build_model_artefact(
    model, scaler, imputer, cfg,
    X_tr, X_te, X_oot,
    y_tr, y_te, y_oot,
    auc_tr, auc_te, auc_oot,
    best_params,
):
    return {
        "model": model,
        "model_version": f"credit_model_{cfg['model_train_date_str'].replace('-', '_')}",
        "preprocessing_transformers": {"scaler": scaler, "imputer": imputer},
        "data_dates": cfg,
        "data_stats": {
            "X_train": X_tr.shape[0],
            "X_test":  X_te.shape[0],
            "X_oot":   X_oot.shape[0],
            "y_train": round(y_tr.mean(), 2),
            "y_test":  round(y_te.mean(), 2),
            "y_oot":   round(y_oot.mean(), 2),
        },
        "results": {
            "auc_train":  auc_tr,
            "auc_test":   auc_te,
            "auc_oot":    auc_oot,
            "gini_train": round(2 * auc_tr  - 1, 3),
            "gini_test":  round(2 * auc_te  - 1, 3),
            "gini_oot":   round(2 * auc_oot - 1, 3),
        },
        "hp_params": best_params,
    }


def save_model_artefact(artefact: Dict[str, Any], save_dir: str = "model_bank/"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, artefact["model_version"] + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"✓ Model artefact saved → {path}")
    return path
