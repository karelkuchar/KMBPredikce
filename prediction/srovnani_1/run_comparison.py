"""
Kolo 1: srovnani feature-setu (univariate/multivariate) x modelu
(Persistence/SeasonalNaive24h/LightGBM/CatBoost/XGBoost) pro autoregresivni
24h predikci Avg.3P[kW] s 15min krokem.

Vystup:
    prediction/srovnani_1/results_summary.csv     - MAE/RMSE/MAPE per (feature_set, model, mode)
    prediction/srovnani_1/results_by_horizon.csv  - MAE/RMSE per krok horizontu 1..96 (jen mode=24h)

Pouziti:
    python3 prediction/srovnani_1/run_comparison.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
FEATURE_MODES = ["univariate", "multivariate"]
ALL_TRAINERS = {**pl.BASELINE_TRAINERS, **pl.MODEL_TRAINERS}


def horizon_breakdown(y_true, y_pred, horizon_step, feature_set, model):
    rows = []
    for h in sorted(set(horizon_step.tolist())):
        mask = horizon_step == h
        if mask.sum() == 0:
            continue
        m = pl.evaluate(y_true[mask], y_pred[mask])
        rows.append(dict(feature_set=feature_set, model=model, horizon_step=int(h),
                          horizon_min=int(h) * pl.RESAMPLE_MIN, n=int(mask.sum()), **m))
    return rows


def main():
    t_start = time.time()
    summary_rows = []
    horizon_rows = []

    for fs_mode in FEATURE_MODES:
        print(f"\n{'='*70}\nFeature set: {fs_mode}\n{'='*70}")
        fs = pl.build_features(fs_mode)
        train, val, test = pl.split_train_val_test(fs.df)
        t0 = len(fs.df) - len(test)
        y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
        print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
              f"features={len(fs.feature_cols)}")

        for name, trainer in ALL_TRAINERS.items():
            t0m = time.time()
            print(f"\n-- {name} ({fs_mode}) --")
            predict_fn = trainer(train, val, fs.feature_cols)

            # 1step (rychla vektorizovana referencni horni mez)
            preds_1step = pl.one_step_predict(predict_fn, test, fs.feature_cols)
            m1 = pl.evaluate(y_true, preds_1step)
            summary_rows.append(dict(feature_set=fs_mode, model=name, mode="1step", **m1))
            print(f"   1step  MAE={m1['mae']}  RMSE={m1['rmse']}  MAPE={m1['mape']}%")

            # 24h (produkcni scenar: reset kazdych 96 kroku)
            preds_24h, hstep_24h = pl.autoregressive_predict(
                predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
            m24 = pl.evaluate(y_true, preds_24h)
            summary_rows.append(dict(feature_set=fs_mode, model=name, mode="24h", **m24))
            print(f"   24h    MAE={m24['mae']}  RMSE={m24['rmse']}  MAPE={m24['mape']}%")
            horizon_rows.extend(horizon_breakdown(y_true, preds_24h, hstep_24h, fs_mode, name))

            # 7d (stress test kumulace chyby)
            preds_7d, hstep_7d = pl.autoregressive_predict(
                predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_7D)
            m7 = pl.evaluate(y_true, preds_7d)
            summary_rows.append(dict(feature_set=fs_mode, model=name, mode="7d", **m7))
            print(f"   7d     MAE={m7['mae']}  RMSE={m7['rmse']}  MAPE={m7['mape']}%")

            print(f"   ({time.time() - t0m:.1f}s)")

    df_summary = pd.DataFrame(summary_rows).sort_values(["mode", "feature_set", "mae"])
    df_horizon = pd.DataFrame(horizon_rows).sort_values(["feature_set", "model", "horizon_step"])

    df_summary.to_csv(OUT_DIR / "results_summary.csv", index=False)
    df_horizon.to_csv(OUT_DIR / "results_by_horizon.csv", index=False)

    print(f"\n{'='*70}\nSOUHRN (mode=24h, serazeno dle MAE):\n{'='*70}")
    print(df_summary[df_summary["mode"] == "24h"].sort_values("mae").to_string(index=False))

    print(f"\nUlozeno:\n  {OUT_DIR / 'results_summary.csv'}\n  {OUT_DIR / 'results_by_horizon.csv'}")
    print(f"\nCelkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
