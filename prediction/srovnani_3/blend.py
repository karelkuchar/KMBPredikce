"""
Kolo 3, dávka 3: blend XGBoost (univariate) predikce se SeasonalNaive24h
baseline (shrinkage smerem ke stabilni kotve, klasicka technika na snizeni
rozptylu dlouho-horizontovych rekurzivnich predikci).

alpha se hleda na validacni sade (14 dni), NE na testu (aby nedoslo k
cherry-pickingu), pak se overi na 30d/60d/staggered.

Pouziti:
    python3 prediction/srovnani_3/blend.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def recursive_on_slice(predict_fn, fs, train_len, n_slice, reset_steps=pl.RESET_24H):
    df_sub = fs.df.iloc[: train_len + n_slice].reset_index(drop=True)
    fs_sub = pl.FeatureSet(name=fs.name, df=df_sub, feature_cols=fs.feature_cols,
                            lag_map=fs.lag_map, exog_cols=fs.exog_cols, time_cols=fs.time_cols)
    preds, _ = pl.autoregressive_predict(predict_fn, fs_sub, n_test=n_slice, reset_steps=reset_steps)
    y_true = df_sub["TARGET"].iloc[train_len:].to_numpy(np.float32)
    return preds, y_true


def main():
    t_start = time.time()
    fs = pl.build_features("univariate")
    train, val, test = pl.split_train_val_test(fs.df)
    train_len, n_val = len(train), len(val)
    print(f"train={train_len}  val={n_val}  test={len(test)}")

    xgb_fn = pl.train_xgboost(train, val, fs.feature_cols)
    seasonal_fn = pl.train_seasonal_naive(train, val, fs.feature_cols)

    # --- alpha search na validaci ---
    preds_xgb_val, y_val = recursive_on_slice(xgb_fn, fs, train_len, n_val)
    preds_seasonal_val, _ = recursive_on_slice(seasonal_fn, fs, train_len, n_val)

    print("\nAlpha search (blend = alpha*XGBoost + (1-alpha)*SeasonalNaive24h) na validaci:")
    best_alpha, best_val_mae = None, None
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend = alpha * preds_xgb_val + (1 - alpha) * preds_seasonal_val
        mae = pl.evaluate(y_val, blend)["mae"]
        print(f"   alpha={alpha:.1f}  val_MAE={mae:.3f}")
        if best_val_mae is None or mae < best_val_mae:
            best_val_mae, best_alpha = mae, alpha
    print(f"-> nejlepsi alpha={best_alpha:.1f} (val_MAE={best_val_mae:.3f})")

    # --- overeni na 30d testu ---
    t0 = len(fs.df) - len(test)
    y_true_test = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
    preds_xgb_test, _ = pl.autoregressive_predict(xgb_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
    preds_seasonal_test, _ = pl.autoregressive_predict(seasonal_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
    blend_test = best_alpha * preds_xgb_test + (1 - best_alpha) * preds_seasonal_test
    m_test = pl.evaluate(y_true_test, blend_test)
    m_test_xgb = pl.evaluate(y_true_test, preds_xgb_test)
    print(f"\n30d test: blend(alpha={best_alpha:.1f}) MAE={m_test['mae']}  vs. pure XGBoost MAE={m_test_xgb['mae']}")

    rows = [
        dict(experiment="blend", alpha=best_alpha, mode="val", **{"mae": best_val_mae}),
        dict(experiment="blend", alpha=best_alpha, mode="test_30d", **m_test),
        dict(experiment="xgboost_pure", alpha=1.0, mode="test_30d", **m_test_xgb),
    ]

    # --- 60d test ---
    pl.TEST_DAYS = 60
    fs60 = pl.build_features("univariate")
    train60, val60, test60 = pl.split_train_val_test(fs60.df)
    xgb_fn60 = pl.train_xgboost(train60, val60, fs60.feature_cols)
    seasonal_fn60 = pl.train_seasonal_naive(train60, val60, fs60.feature_cols)
    t060 = len(fs60.df) - len(test60)
    y_true_60 = fs60.df["TARGET"].iloc[t060:].to_numpy(np.float32)
    preds_xgb_60, _ = pl.autoregressive_predict(xgb_fn60, fs60, n_test=len(test60), reset_steps=pl.RESET_24H)
    preds_seasonal_60, _ = pl.autoregressive_predict(seasonal_fn60, fs60, n_test=len(test60), reset_steps=pl.RESET_24H)
    blend_60 = best_alpha * preds_xgb_60 + (1 - best_alpha) * preds_seasonal_60
    m60 = pl.evaluate(y_true_60, blend_60)
    m60_xgb = pl.evaluate(y_true_60, preds_xgb_60)
    print(f"60d test: blend(alpha={best_alpha:.1f}) MAE={m60['mae']}  vs. pure XGBoost MAE={m60_xgb['mae']}")
    rows.append(dict(experiment="blend", alpha=best_alpha, mode="test_60d", **m60))
    rows.append(dict(experiment="xgboost_pure", alpha=1.0, mode="test_60d", **m60_xgb))

    pd.DataFrame(rows).to_csv(OUT_DIR / "blend_results.csv", index=False)
    print(f"\nUlozeno: {OUT_DIR / 'blend_results.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
