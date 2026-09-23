"""
Kolo 3, dávka 2: ověření robustnosti univariate_v2 + XGBoost (nadějný
kandidát z dávky 1, 21.36 kW na 30denním okně) - 60denní okno + staggered eval,
stejná metodologie jako u předchozích kol.

Pouziti:
    python3 prediction/srovnani_3/verify_v2.py
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

    # --- 60denni okno ---
    pl.TEST_DAYS = 60
    fs = pl.build_features("univariate_v2")
    train, val, test = pl.split_train_val_test(fs.df)
    t0 = len(fs.df) - len(test)
    y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
    print(f"60d test: train={len(train)}  val={len(val)}  test={len(test)}")

    predict_fn = pl.train_xgboost(train, val, fs.feature_cols)
    preds_24h, hstep = pl.autoregressive_predict(predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
    m24 = pl.evaluate(y_true, preds_24h)
    print(f"univariate_v2 + XGBoost, 24h(60d) MAE={m24['mae']}  RMSE={m24['rmse']}  MAPE={m24['mape']}%")
    rows.append(dict(feature_set="univariate_v2", model="XGBoost", mode="24h", test_days=60, **m24))

    # --- staggered (fazove-neutralni) eval na 30denním oknu, stejny model ---
    pl.TEST_DAYS = 30
    fs30 = pl.build_features("univariate_v2")
    train30, val30, test30 = pl.split_train_val_test(fs30.df)
    t030 = len(fs30.df) - len(test30)
    predict_fn30 = pl.train_xgboost(train30, val30, fs30.feature_cols)

    n_total = len(fs30.df)
    start_indices = list(range(t030, n_total - 96 + 1, 4))
    df_stag = pl.staggered_recursive_predict(predict_fn30, fs30, start_indices, horizon=96)
    m_stag = pl.evaluate(df_stag["y_true"].to_numpy(), df_stag["y_pred"].to_numpy())
    print(f"univariate_v2 + XGBoost, staggered(30d) MAE={m_stag['mae']}  ({len(start_indices)} startu)")
    rows.append(dict(feature_set="univariate_v2", model="XGBoost", mode="staggered_30d", test_days=30, **m_stag))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "verify_v2_results.csv", index=False)
    print(f"\nUlozeno: {OUT_DIR / 'verify_v2_results.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
