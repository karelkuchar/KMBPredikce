"""
Kolo 3, davka 4: budova je univerzitni pracoviste (VUT) - spotreba se muze
ridit semestrem/prazdninami, ne jen dnem v tydnu (info od uzivatele). Testuje
novy feature-set 'univariate_semester' (+ `is_break`, priblizny akademicky
kalendar - viz pipeline.py::is_academic_break, PRIBLIZNE datumy).

Rovnou testuje na 30d I 60d I staggered (poucení z davky 1 - univariate_v2
vypadal dobre jen na 30d a neprosel), aby se nemuselo cekat na dalsi kolo.

Pouziti:
    python3 prediction/srovnani_3/semester.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def eval_30_60_staggered(fs_mode):
    rows = []

    # --- 30d ---
    fs = pl.build_features(fs_mode)
    train, val, test = pl.split_train_val_test(fs.df)
    t0 = len(fs.df) - len(test)
    y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
    predict_fn = pl.train_xgboost(train, val, fs.feature_cols)
    preds_30, _ = pl.autoregressive_predict(predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
    m30 = pl.evaluate(y_true, preds_30)
    print(f"  {fs_mode:22s} 30d  MAE={m30['mae']}")
    rows.append(dict(feature_set=fs_mode, mode="30d", **m30))

    # --- staggered (30d test window, same model) ---
    n_total = len(fs.df)
    start_indices = list(range(t0, n_total - 96 + 1, 4))
    df_stag = pl.staggered_recursive_predict(predict_fn, fs, start_indices, horizon=96)
    m_stag = pl.evaluate(df_stag["y_true"].to_numpy(), df_stag["y_pred"].to_numpy())
    print(f"  {fs_mode:22s} staggered  MAE={m_stag['mae']}  ({len(start_indices)} startu)")
    rows.append(dict(feature_set=fs_mode, mode="staggered_30d", **m_stag))

    # --- 60d ---
    pl.TEST_DAYS = 60
    fs60 = pl.build_features(fs_mode)
    train60, val60, test60 = pl.split_train_val_test(fs60.df)
    t060 = len(fs60.df) - len(test60)
    y_true60 = fs60.df["TARGET"].iloc[t060:].to_numpy(np.float32)
    predict_fn60 = pl.train_xgboost(train60, val60, fs60.feature_cols)
    preds_60, _ = pl.autoregressive_predict(predict_fn60, fs60, n_test=len(test60), reset_steps=pl.RESET_24H)
    m60 = pl.evaluate(y_true60, preds_60)
    print(f"  {fs_mode:22s} 60d  MAE={m60['mae']}")
    rows.append(dict(feature_set=fs_mode, mode="60d", **m60))
    pl.TEST_DAYS = 30

    return rows


def main():
    t_start = time.time()
    all_rows = []
    for fs_mode in ["univariate", "univariate_semester"]:
        print(f"\n{'='*70}\n{fs_mode} + XGBoost\n{'='*70}")
        all_rows += eval_30_60_staggered(fs_mode)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT_DIR / "semester_results.csv", index=False)
    print(f"\n{'='*70}\nSOUHRN:\n{'='*70}")
    print(df_out.pivot(index="mode", columns="feature_set", values="mae").to_string())
    print(f"\nUlozeno: {OUT_DIR / 'semester_results.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
