"""
Kolo 3, dávka 1: pokus o vylepšení nad univariate+XGBoost baseline (22.56 kW,
prediction/EXPERIMENT_LOG.md "Oprava kritického bufferového bugu").

Testuje:
  - univariate (baseline, reprodukce)
  - univariate_v2 (+ lag_2week, lag_3week - silnejsi tydenni signal, viz pipeline.py)
  - ensemble: prosty prumer predikci XGBoost + CatBoost (univariate), pocitany
    post-hoc z jejich jiz natrenovanych 24h rekurzivnich predikci

Vystup:
    prediction/srovnani_3/results_summary.csv

Pouziti:
    python3 prediction/srovnani_3/run_comparison.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main():
    t_start = time.time()
    rows = []

    # --- univariate baseline (reprodukce) + univariate_v2 ---
    for fs_mode in ["univariate", "univariate_v2"]:
        print(f"\n{'='*70}\nFeature set: {fs_mode}\n{'='*70}")
        fs = pl.build_features(fs_mode)
        train, val, test = pl.split_train_val_test(fs.df)
        t0 = len(fs.df) - len(test)
        y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
        print(f"train={len(train)}  val={len(val)}  test={len(test)}  features={len(fs.feature_cols)}")

        preds_by_model = {}
        for name, trainer in [("XGBoost", pl.train_xgboost), ("CatBoost", pl.train_catboost)]:
            t0m = time.time()
            predict_fn = trainer(train, val, fs.feature_cols)
            preds_24h, _ = pl.autoregressive_predict(predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
            preds_by_model[name] = preds_24h
            m24 = pl.evaluate(y_true, preds_24h)
            rows.append(dict(feature_set=fs_mode, model=name, mode="24h", **m24))
            print(f"   {name:10s} 24h MAE={m24['mae']}  RMSE={m24['rmse']}  MAPE={m24['mape']}%  ({time.time()-t0m:.1f}s)")

        if fs_mode == "univariate":
            ens = (preds_by_model["XGBoost"] + preds_by_model["CatBoost"]) / 2.0
            m_ens = pl.evaluate(y_true, ens)
            rows.append(dict(feature_set=fs_mode, model="Ensemble_XGB_Cat", mode="24h", **m_ens))
            print(f"   {'Ensemble':10s} 24h MAE={m_ens['mae']}  RMSE={m_ens['rmse']}  MAPE={m_ens['mape']}%")

    df_out = pd.DataFrame(rows).sort_values("mae")
    df_out.to_csv(OUT_DIR / "results_summary.csv", index=False)
    print(f"\n{'='*70}\nSOUHRN (serazeno dle MAE):\n{'='*70}")
    print(df_out.to_string(index=False))
    print(f"\nUlozeno: {OUT_DIR / 'results_summary.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
