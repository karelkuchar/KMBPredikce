"""
Kolo 2, krok 4: overeni na delsim test okne (60 dni misto 30).

Testuje robustnost vitezne kombinace (multivariate + XGBoost, vychozi i
vyladene hyperparametry z tune.py) a par baseline/srovnavacich modelu na
2x delsim testovacim okne, aby 30-denni vysledek nebyl nahoda dana volbou
konkretniho useku.

Vystup (nazev dle feature-setu):
    prediction/srovnani_2/results_summary_test60[_<feature-set>].csv

Pouziti:
    python3 prediction/srovnani_2/test60.py [univariate|multivariate|multivariate_plus]
    (bez argumentu = multivariate, zpetne kompatibilni jmeno souboru)
"""

import json
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

with open(OUT_DIR / f"tuning_best{SUFFIX}.json") as f:
    TUNED_PARAMS = json.load(f)


def main():
    t_start = time.time()
    pl.TEST_DAYS = 60  # prodlouzeno z vychoziho 30

    fs = pl.build_features(FS_MODE)
    train, val, test = pl.split_train_val_test(fs.df)
    t0 = len(fs.df) - len(test)
    y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
    print(f"TEST_DAYS={pl.TEST_DAYS}  train={len(train)}  val={len(val)}  test={len(test)}")

    combos = [
        ("XGBoost_default", pl.train_xgboost, None),
        ("XGBoost_tuned", pl.train_xgboost, TUNED_PARAMS["XGBoost"]),
        ("CatBoost_default", pl.train_catboost, None),
        ("CatBoost_tuned", pl.train_catboost, TUNED_PARAMS["CatBoost"]),
        ("Persistence", pl.train_persistence, None),
        ("SeasonalNaive24h", pl.train_seasonal_naive, None),
    ]

    rows = []
    for name, trainer, params in combos:
        t0m = time.time()
        if params is not None:
            predict_fn = trainer(train, val, fs.feature_cols, params=params)
        else:
            predict_fn = trainer(train, val, fs.feature_cols)

        preds_24h, _ = pl.autoregressive_predict(predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
        m24 = pl.evaluate(y_true, preds_24h)
        rows.append(dict(feature_set=FS_MODE, model=name, mode="24h", test_days=60, **m24))
        print(f"{name:20s} 24h(60d) MAE={m24['mae']}  RMSE={m24['rmse']}  MAPE={m24['mape']}%  ({time.time()-t0m:.1f}s)")

    df = pd.DataFrame(rows).sort_values("mae")
    df.to_csv(OUT_DIR / f"results_summary_test60{SUFFIX}.csv", index=False)
    print(f"\nUlozeno: {OUT_DIR / f'results_summary_test60{SUFFIX}.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
