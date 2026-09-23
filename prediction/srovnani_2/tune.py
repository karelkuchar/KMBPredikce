"""
Kolo 2, krok 3: hyperparameter tuning.

Feature-set: 'multivariate' (kolo 1/2 vitez - 'multivariate_plus' vysel v
run_comparison.py hur v rekurzivnim modu, viz EXPERIMENT_LOG.md).

Duvod, proc netunit podle 1step validacni MAE (jak to delal puvodni
referencni skript): 1step MAE mezi feature-sety/modely je temer nerozlisitelne
(~6.0-6.3 kW pro vsechny), ale 24h rekurzivni MAE se lisi az 2x (28-51 kW) -
1step neodrazi, jak model chybuje pri kumulaci pres 96 kroku. Tuning proto
probiha primo na 24h rekurzivni MAE na validacni sade (14 dni, reset_steps=96,
= stejny rezim jako produkce), ne na 1step.

Vystup (nazev souboru dle feature-setu, napr. pro univariate):
    prediction/srovnani_2/tuning_trials_univariate.csv
    prediction/srovnani_2/tuning_best_univariate.json
    prediction/srovnani_2/results_summary_tuned_univariate.csv

Pouziti:
    python3 prediction/srovnani_2/tune.py [univariate|multivariate|multivariate_plus]
    (bez argumentu = multivariate, vystupni soubory bez suffixu - zachovava
    puvodni jmena pro zpetnou kompatibilitu)
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
FS_MODE = sys.argv[1] if len(sys.argv) > 1 else "multivariate"
SUFFIX = "" if FS_MODE == "multivariate" else f"_{FS_MODE}"
N_TRIALS = 25
RNG = random.Random(pl.SEED)

XGB_GRID = dict(
    learning_rate=[0.02, 0.03, 0.05, 0.08, 0.1],
    max_depth=[4, 5, 6, 8, 10],
    subsample=[0.6, 0.7, 0.8, 0.9, 1.0],
    colsample_bytree=[0.6, 0.7, 0.8, 0.9, 1.0],
    min_child_weight=[1, 3, 5, 10],
    reg_alpha=[0.0, 0.1, 0.5, 1.0],
    reg_lambda=[0.5, 1.0, 2.0, 5.0],
)

CAT_GRID = dict(
    learning_rate=[0.02, 0.03, 0.05, 0.08, 0.1],
    depth=[4, 5, 6, 8, 10],
    l2_leaf_reg=[1.0, 3.0, 5.0, 10.0],
    random_strength=[0.5, 1.0, 2.0, 5.0],
)


def sample_params(grid):
    return {k: RNG.choice(v) for k, v in grid.items()}


def eval_on_slice(predict_fn, fs, train_len, n_val):
    """24h-rekurzivni beh presne na useku [train_len, train_len+n_val) fs.df
    (autoregressive_predict pocita t0 = len(df)-n_test, proto orizneme df na
    train_len+n_val radku, aby beh zacal presne za train a ne az na konci fs.df)."""
    df_sub = fs.df.iloc[: train_len + n_val].reset_index(drop=True)
    fs_sub = pl.FeatureSet(name=fs.name, df=df_sub, feature_cols=fs.feature_cols,
                            lag_map=fs.lag_map, exog_cols=fs.exog_cols, time_cols=fs.time_cols)
    preds, _ = pl.autoregressive_predict(predict_fn, fs_sub, n_test=n_val, reset_steps=pl.RESET_24H)
    y_true = df_sub["TARGET"].iloc[train_len:].to_numpy(np.float32)
    return pl.evaluate(y_true, preds)["mae"]


def run_search(model_name, trainer_fn, grid, fs, train, val):
    train_len = len(train)
    n_val = len(val)
    trials = []
    best = None
    for i in range(N_TRIALS):
        params = sample_params(grid)
        t0 = time.time()
        try:
            predict_fn = trainer_fn(train, val, fs.feature_cols, params)
            mae = eval_on_slice(predict_fn, fs, train_len, n_val)
        except Exception as e:  # pragma: no cover - defensive, log and skip bad combo
            print(f"   trial {i:02d} FAILED ({e}) params={params}")
            continue
        dt = time.time() - t0
        trials.append(dict(model=model_name, trial=i, val_mae_24h=round(mae, 5), seconds=round(dt, 1), **params))
        print(f"   trial {i:02d}/{N_TRIALS}  val_MAE_24h={mae:.3f}  ({dt:.1f}s)  {params}")
        if best is None or mae < best["val_mae_24h"]:
            best = trials[-1]
    return trials, best


def main():
    t_start = time.time()
    fs = pl.build_features(FS_MODE)
    train, val, test = pl.split_train_val_test(fs.df)
    print(f"feature_set={FS_MODE}  train={len(train)}  val={len(val)}  test={len(test)}")

    all_trials = []
    best_params = {}

    print(f"\n{'='*70}\nXGBoost random search ({N_TRIALS} trials)\n{'='*70}")
    trials, best = run_search("XGBoost", pl.train_xgboost, XGB_GRID, fs, train, val)
    all_trials += trials
    best_params["XGBoost"] = {k: v for k, v in best.items() if k not in ("model", "trial", "val_mae_24h", "seconds")}
    print(f"-> best XGBoost val_MAE_24h={best['val_mae_24h']}  params={best_params['XGBoost']}")

    print(f"\n{'='*70}\nCatBoost random search ({N_TRIALS} trials)\n{'='*70}")
    trials, best = run_search("CatBoost", pl.train_catboost, CAT_GRID, fs, train, val)
    all_trials += trials
    best_params["CatBoost"] = {k: v for k, v in best.items() if k not in ("model", "trial", "val_mae_24h", "seconds")}
    print(f"-> best CatBoost val_MAE_24h={best['val_mae_24h']}  params={best_params['CatBoost']}")

    pd.DataFrame(all_trials).to_csv(OUT_DIR / f"tuning_trials{SUFFIX}.csv", index=False)
    with open(OUT_DIR / f"tuning_best{SUFFIX}.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # finalni over na testu s vitaznymi hyperparametry (retrain na train+val jako trainer_fn dela: train pro fit, val pro early stopping)
    print(f"\n{'='*70}\nFinalni test s vitaznymi hyperparametry (feature_set={FS_MODE})\n{'='*70}")
    y_true_test = fs.df["TARGET"].iloc[len(fs.df) - len(test):].to_numpy(np.float32)
    final_rows = []
    for model_name, trainer_fn in [("XGBoost", pl.train_xgboost), ("CatBoost", pl.train_catboost)]:
        predict_fn = trainer_fn(train, val, fs.feature_cols, best_params[model_name])
        preds_1step = pl.one_step_predict(predict_fn, test, fs.feature_cols)
        m1 = pl.evaluate(y_true_test, preds_1step)
        preds_24h, _ = pl.autoregressive_predict(predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
        m24 = pl.evaluate(y_true_test, preds_24h)
        preds_7d, _ = pl.autoregressive_predict(predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_7D)
        m7 = pl.evaluate(y_true_test, preds_7d)
        print(f"{model_name} (tuned): 1step MAE={m1['mae']}  24h MAE={m24['mae']}  7d MAE={m7['mae']}")
        final_rows.append(dict(feature_set=FS_MODE, model=model_name + "_tuned", mode="1step", **m1))
        final_rows.append(dict(feature_set=FS_MODE, model=model_name + "_tuned", mode="24h", **m24))
        final_rows.append(dict(feature_set=FS_MODE, model=model_name + "_tuned", mode="7d", **m7))

    pd.DataFrame(final_rows).to_csv(OUT_DIR / f"results_summary_tuned{SUFFIX}.csv", index=False)
    print(f"\nUlozeno:\n  {OUT_DIR / f'tuning_trials{SUFFIX}.csv'}\n  {OUT_DIR / f'tuning_best{SUFFIX}.json'}"
          f"\n  {OUT_DIR / f'results_summary_tuned{SUFFIX}.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
