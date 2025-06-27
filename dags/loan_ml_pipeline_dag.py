from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import pendulum

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# ── project utilities ─────────────────────────────────────────────────────
from utils import data_io, modelling, inference, monitoring

# -----------------------------------------------------------------------------
# 1. Task wrappers
# -----------------------------------------------------------------------------
def load_data_task(ds, **_):
    date_from = "2023-01-01"
    date_to   = ds

    X, y = data_io.load_dataset(
        date_from=date_from,
        date_to=date_to,
        label_col="label",
        filter_leakage=True,
    )

    if X.empty:
        raise ValueError(f"No rows between {date_from} and {date_to}")

    print(f"[✓] Dataset {date_from} → {date_to}: "
          f"{X.shape[0]:,} rows, {X.shape[1]} features.")


def training_task(ds, **_):
    inf_dt         = pd.to_datetime(ds)                         
    oot_end_dt     = (inf_dt - relativedelta(months=1)).replace(day=1)  
    oot_start_dt   = (oot_end_dt - relativedelta(months=2))             
    train_end_dt   = oot_start_dt - timedelta(days=1)                   
    train_start_dt = (oot_start_dt - relativedelta(months=18)).replace(day=1)  

    cfg = {
        "train_test_start_date": train_start_dt.strftime("%Y-%m-%d"),
        "train_test_end_date"  : train_end_dt.strftime("%Y-%m-%d"),
        "oot_start_date"       : oot_start_dt.strftime("%Y-%m-%d"),
        "oot_end_date"         : (oot_end_dt + MonthEnd(0)).strftime("%Y-%m-%d"),
        "train_test_ratio"     : 0.8,
        "model_train_date_str" : ds,
    }
    print("Date config for this run:", cfg)

    X_raw, y = data_io.load_dataset(
        date_from=cfg["train_test_start_date"],
        date_to  =cfg["oot_end_date"],
        label_col="label",
        filter_leakage=True,
    )
    df_raw = X_raw.copy()
    df_raw["label"] = y

    (X_tr_raw, X_te_raw, X_oot_raw, y_tr, y_te, y_oot) = modelling.split_data(df_raw, cfg)

    X_tr, imp = modelling.preprocess_features(X_tr_raw)
    X_te, _   = modelling.preprocess_features(X_te_raw,  imp)
    X_oot, _  = modelling.preprocess_features(X_oot_raw, imp)

    scaler, X_tr_s, X_te_s, X_oot_s = modelling.scale_features(X_tr, X_te, X_oot)

    model, best_params, _ = modelling.train_xgboost_with_search(X_tr_s, y_tr, n_iter=10)

    auc_tr, _ = modelling.evaluate_model(model, X_tr_s,  y_tr)
    auc_te, _ = modelling.evaluate_model(model, X_te_s,  y_te)
    auc_oot,_ = modelling.evaluate_model(model, X_oot_s, y_oot)

    artefact = modelling.build_model_artefact(
        model, scaler, imp, cfg,
        X_tr_s, X_te_s, X_oot_s,
        y_tr, y_te, y_oot,
        auc_tr, auc_te, auc_oot,
        best_params,
    )
    modelling.save_model_artefact(artefact)

def inference_task(ds, **_):
    inference.run_inference(ds)

def monitoring_task(ds, **_):
    monitoring.run_monitoring()

# -----------------------------------------------------------------------------
# 2. DAG definition
# -----------------------------------------------------------------------------
default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    dag_id            = "loan_credit_ml",
    start_date        = pendulum.datetime(2024, 10, 1, 3, 0),
    schedule_interval = "0 3 1 * *",     # monthly at 03:00 on the 1st
    catchup           = True,
    max_active_runs   = 1,
    default_args      = default_args,
    tags              = ["loan", "ml"],
)

with dag:
    start  = EmptyOperator(task_id="start")

    load_data = PythonOperator(
        task_id         = "load_data",
        python_callable = load_data_task,
        op_kwargs       = {"ds": "{{ ds }}"},
    )

    train = PythonOperator(
        task_id         = "training",
        python_callable = training_task,
        op_kwargs       = {"ds": "{{ ds }}"},
    )

    infer = PythonOperator(
        task_id         = "inference",
        python_callable = inference_task,
        op_kwargs       = {"ds": "{{ ds }}"},
    )

    monitor = PythonOperator(
        task_id         = "monitoring",
        python_callable = monitoring_task,
        op_kwargs       = {"ds": "{{ ds }}"},
    )

    end = EmptyOperator(task_id="end")

    # ── linear flow ───────────────────────────────────────────────────────
    start >> load_data >> train >> infer >> monitor >> end
