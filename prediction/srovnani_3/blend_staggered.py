"""
Kolo 3, dávka 3b: staggered (fázově-neutrální) ověření blendu
alpha*XGBoost + (1-alpha)*SeasonalNaive24h, alpha=0.6 z blend.py.

Pouziti:
    python3 prediction/srovnani_3/blend_staggered.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
ALPHA = 0.6


def main():
    t_start = time.time()
    fs = pl.build_features("univariate")
    train, val, test = pl.split_train_val_test(fs.df)
    t0 = len(fs.df) - len(test)
    n_total = len(fs.df)

    xgb_fn = pl.train_xgboost(train, val, fs.feature_cols)
    seasonal_fn = pl.train_seasonal_naive(train, val, fs.feature_cols)

    start_indices = list(range(t0, n_total - 96 + 1, 4))
    df_xgb = pl.staggered_recursive_predict(xgb_fn, fs, start_indices, horizon=96)
    df_seasonal = pl.staggered_recursive_predict(seasonal_fn, fs, start_indices, horizon=96)

    assert (df_xgb["global_idx"].to_numpy() == df_seasonal["global_idx"].to_numpy()).all()
    y_true = df_xgb["y_true"].to_numpy()
    blend_pred = ALPHA * df_xgb["y_pred"].to_numpy() + (1 - ALPHA) * df_seasonal["y_pred"].to_numpy()

    m_blend = pl.evaluate(y_true, blend_pred)
    m_xgb = pl.evaluate(y_true, df_xgb["y_pred"].to_numpy())
    m_seasonal = pl.evaluate(y_true, df_seasonal["y_pred"].to_numpy())
    print(f"staggered ({len(start_indices)} startu):")
    print(f"  XGBoost pure       MAE={m_xgb['mae']}")
    print(f"  SeasonalNaive24h   MAE={m_seasonal['mae']}")
    print(f"  Blend (alpha={ALPHA}) MAE={m_blend['mae']}")

    pd.DataFrame([
        dict(model="XGBoost_pure", **m_xgb),
        dict(model="SeasonalNaive24h", **m_seasonal),
        dict(model=f"Blend_alpha{ALPHA}", **m_blend),
    ]).to_csv(OUT_DIR / "blend_staggered_results.csv", index=False)
    print(f"\nUlozeno: {OUT_DIR / 'blend_staggered_results.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
