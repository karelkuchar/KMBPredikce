"""
Kolo 3, davka 5: kombinace univariate_semester (is_break) s existujicim
nasazenym blendem: blend(alpha*XGBoost_semester + (1-alpha)*SeasonalNaive24h).
Zjistuje, jestli mala zlepseni z davky 4 (univariate_semester samotne) drzi i
po zkombinovani s blendem, a jestli spolu prekonaji nasazenych 21.14 kW.

Pouziti:
    python3 prediction/srovnani_3/blend_semester.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def blend_eval(fs, xgb_fn, seasonal_fn, n_test, reset_steps, t0=None):
    preds_xgb, _ = pl.autoregressive_predict(xgb_fn, fs, n_test=n_test, reset_steps=reset_steps)
    preds_seasonal, _ = pl.autoregressive_predict(seasonal_fn, fs, n_test=n_test, reset_steps=reset_steps)
    return preds_xgb, preds_seasonal


def main():
    t_start = time.time()
    fs = pl.build_features("univariate_semester")
    train, val, test = pl.split_train_val_test(fs.df)
    train_len, n_val = len(train), len(val)
    print(f"train={train_len}  val={n_val}  test={len(test)}")

    xgb_fn = pl.train_xgboost(train, val, fs.feature_cols)
    seasonal_fn = pl.train_seasonal_naive(train, val, fs.feature_cols)

    # --- alpha search na validaci ---
    df_val_sub = fs.df.iloc[: train_len + n_val].reset_index(drop=True)
    fs_val_sub = pl.FeatureSet(name=fs.name, df=df_val_sub, feature_cols=fs.feature_cols,
                                lag_map=fs.lag_map, exog_cols=fs.exog_cols, time_cols=fs.time_cols)
    y_val = df_val_sub["TARGET"].iloc[train_len:].to_numpy(np.float32)
    preds_xgb_val, _ = pl.autoregressive_predict(xgb_fn, fs_val_sub, n_test=n_val, reset_steps=pl.RESET_24H)
    preds_seasonal_val, _ = pl.autoregressive_predict(seasonal_fn, fs_val_sub, n_test=n_val, reset_steps=pl.RESET_24H)

    print("\nAlpha search na validaci:")
    best_alpha, best_mae = None, None
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend = alpha * preds_xgb_val + (1 - alpha) * preds_seasonal_val
        mae = pl.evaluate(y_val, blend)["mae"]
        print(f"  alpha={alpha:.1f}  val_MAE={mae:.3f}")
        if best_mae is None or mae < best_mae:
            best_mae, best_alpha = mae, round(float(alpha), 2)
    print(f"-> nejlepsi alpha={best_alpha} (val_MAE={best_mae:.3f})")

    rows = []

    # --- 30d test ---
    t0 = len(fs.df) - len(test)
    y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
    preds_xgb_30, preds_seasonal_30 = blend_eval(fs, xgb_fn, seasonal_fn, len(test), pl.RESET_24H)
    blend_30 = best_alpha * preds_xgb_30 + (1 - best_alpha) * preds_seasonal_30
    m30 = pl.evaluate(y_true, blend_30)
    print(f"\n30d blend MAE={m30['mae']}  (nasazeny blend bez semester: 21.135)")
    rows.append(dict(mode="30d", alpha=best_alpha, **m30))

    # --- staggered ---
    n_total = len(fs.df)
    start_indices = list(range(t0, n_total - 96 + 1, 4))
    df_xgb_stag = pl.staggered_recursive_predict(xgb_fn, fs, start_indices, horizon=96)
    df_seasonal_stag = pl.staggered_recursive_predict(seasonal_fn, fs, start_indices, horizon=96)
    y_true_stag = df_xgb_stag["y_true"].to_numpy()
    blend_stag = best_alpha * df_xgb_stag["y_pred"].to_numpy() + (1 - best_alpha) * df_seasonal_stag["y_pred"].to_numpy()
    m_stag = pl.evaluate(y_true_stag, blend_stag)
    print(f"staggered blend MAE={m_stag['mae']}  (nasazeny blend bez semester: 20.830)")
    rows.append(dict(mode="staggered_30d", alpha=best_alpha, **m_stag))

    # --- 60d ---
    pl.TEST_DAYS = 60
    fs60 = pl.build_features("univariate_semester")
    train60, val60, test60 = pl.split_train_val_test(fs60.df)
    xgb_fn60 = pl.train_xgboost(train60, val60, fs60.feature_cols)
    seasonal_fn60 = pl.train_seasonal_naive(train60, val60, fs60.feature_cols)
    t060 = len(fs60.df) - len(test60)
    y_true60 = fs60.df["TARGET"].iloc[t060:].to_numpy(np.float32)
    preds_xgb_60, preds_seasonal_60 = blend_eval(fs60, xgb_fn60, seasonal_fn60, len(test60), pl.RESET_24H)
    blend_60 = best_alpha * preds_xgb_60 + (1 - best_alpha) * preds_seasonal_60
    m60 = pl.evaluate(y_true60, blend_60)
    print(f"60d blend MAE={m60['mae']}  (nasazeny blend bez semester: 21.403)")
    rows.append(dict(mode="60d", alpha=best_alpha, **m60))

    pd.DataFrame(rows).to_csv(OUT_DIR / "blend_semester_results.csv", index=False)
    print(f"\nUlozeno: {OUT_DIR / 'blend_semester_results.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
