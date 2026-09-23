"""
Natrenuje XGBoost s VYCHOZIMI hyperparametry (feature-set 'univariate': jen
lagy/rolling statistiky/kalendarni featury cile, bez exogennich veliciny -
viz prediction/EXPERIMENT_LOG.md "Oprava bufferoveho bugu"), pak najde
nejlepsi blend-vahu alpha pro `common.blend_autoregressive_predict`
(blend = alpha*XGBoost + (1-alpha)*SeasonalNaive24h, kazdy jako nezavisla
rekurze - viz "Kolo 3" v EXPERIMENT_LOG.md - shrinkage smerem k sezonni
kotve snizuje rozptyl 24h rekurzivni predikce o dalsich ~6-8 %). Ulozi
vsechny vystupy behu do vlastni slozky pojmenovane datem/casem behu:

    prediction_final/runs/<YYYYMMDD_HHMMSS>/model.json
    prediction_final/runs/<YYYYMMDD_HHMMSS>/best_hyperparams.json
    prediction_final/runs/<YYYYMMDD_HHMMSS>/metadata.json
    prediction_final/runs/<YYYYMMDD_HHMMSS>/train.log
    prediction_final/runs/latest.json   - ukazatel na nejnovejsi beh (ctou predict.py/test/run_test.py)

Kazdy beh tak zustava samostatny a slozka prediction_final/ nezustava
zaneseana prepisovanymi soubory z ruznych behu.

**Proc bez hyperparameter-tuningu XGBoostu:** v kole 3 bylo overeno (viz
EXPERIMENT_LOG.md), ze kombinace "25-trial hyperparameter search + alpha
search" na stejnem 14-denním validacnim okne prekombinovava a preplacne
overfituje - vysledny model mel skvely val MAE (18.6 kW), ale test MAE
27.7 kW (mnohem hur nez cisty vychozi XGBoost, natoz nez spravne
zvalidovany blend). Vychozi hyperparametry + jediny (alpha) search jsou
proto bezpecnejsi a shoduji se s nezavisle overenou metodikou v
prediction/srovnani_3/blend.py (robustni na 30d/60d/staggered eval).

Alpha search cili primo na 24h rekurzivni MAE na validacni sade (14 dni).

Delku testovaciho okna nastav primo zmenou TEST_DAYS nize (vychozi 30 dni,
viz kolo 2/3 - 30d/60d/staggered overeni pro srovnani ruznych delek).

Pouziti:
    python3 prediction_final/train.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

import common

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"

TEST_DAYS = common.TEST_DAYS  # delka drzeneho testovaciho okna ve dnech - zmen zde pro jiny beh

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.9, objective="reg:absoluteerror",
    random_state=common.SEED, early_stopping_rounds=150, eval_metric="mae",
)


class Tee:
    """Zrcadli print() vystup na stdout i do souboru - aby run.log obsahoval
    kompletni prubeh behu i pri primem spusteni bez shellove presmerovani."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def fit_xgb(train, val, feature_cols):
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[feature_cols].to_numpy(np.float32), train["TARGET"].to_numpy(np.float32),
              eval_set=[(val[feature_cols].to_numpy(np.float32), val["TARGET"].to_numpy(np.float32))],
              verbose=False)
    return model


def main():
    common.TEST_DAYS = TEST_DAYS

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)

    log_file = open(run_dir / "train.log", "w")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_file)
    try:
        _run(run_dir)
    finally:
        sys.stdout = real_stdout
        log_file.close()


def _run(run_dir: Path):
    t_start = time.time()
    print(f"TEST_DAYS={TEST_DAYS}")
    df = common.load_data()
    fs = common.build_features(df)
    train, val, test = common.split_train_val_test(fs.df)
    train_len, n_val = len(train), len(val)
    print(f"train={train_len}  val={n_val}  test={len(test)}  features={len(fs.feature_cols)}")

    print(f"\nTrenink XGBoost (vychozi hyperparametry): {XGB_PARAMS}")
    xgb_model = fit_xgb(train, val, fs.feature_cols)

    # --- alpha search na validaci: blend = alpha*XGBoost + (1-alpha)*SeasonalNaive24h ---
    df_val_sub = fs.df.iloc[: train_len + n_val].reset_index(drop=True)
    fs_val_sub = common.FeatureSet(df=df_val_sub, feature_cols=fs.feature_cols, lag_map=fs.lag_map)
    y_val = df_val_sub["TARGET"].iloc[train_len:].to_numpy(np.float32)
    preds_xgb_val = common.autoregressive_predict(xgb_model, fs_val_sub, n_test=n_val, reset_steps=common.RESET_24H)
    seasonal_model = common.SeasonalNaiveModel(fs.feature_cols)
    preds_seasonal_val = common.autoregressive_predict(seasonal_model, fs_val_sub, n_test=n_val, reset_steps=common.RESET_24H)

    print(f"\nAlpha search (blend = alpha*XGBoost + (1-alpha)*SeasonalNaive24h, kazdy jako"
          f" nezavisla rekurze) na validaci:")
    best_alpha, best_alpha_mae = None, None
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend = alpha * preds_xgb_val + (1 - alpha) * preds_seasonal_val
        mae = common.evaluate(y_val, blend)["mae"]
        print(f"  alpha={alpha:.1f}  val_MAE_24h={mae:.3f}")
        if best_alpha_mae is None or mae < best_alpha_mae:
            best_alpha_mae, best_alpha = mae, round(float(alpha), 2)
    print(f"-> nejlepsi alpha={best_alpha} (val_MAE={best_alpha_mae:.3f})")

    # finalni test na drzenem test okne
    t0 = len(fs.df) - len(test)
    y_true_test = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
    preds_test = common.blend_autoregressive_predict(xgb_model, fs, n_test=len(test), alpha=best_alpha,
                                                       reset_steps=common.RESET_24H)
    m_test = common.evaluate(y_true_test, preds_test)
    print(f"Test 24h MAE={m_test['mae']} kW  RMSE={m_test['rmse']} kW  MAPE={m_test['mape']}%  (blend alpha={best_alpha})")

    xgb_model.save_model(str(run_dir / "model.json"))

    with open(run_dir / "best_hyperparams.json", "w") as f:
        json.dump({**XGB_PARAMS, "blend_alpha": best_alpha, "blend_val_mae_24h_kW": round(best_alpha_mae, 3)}, f, indent=2)

    metadata = dict(
        feature_cols=fs.feature_cols,
        lag_map=fs.lag_map,
        time_cols=common.TIME_COLS,
        target_col=common.TARGET_COL,
        model_start=str(common.MODEL_START),
        test_days=TEST_DAYS,
        blend_alpha=best_alpha,
        test_mae_24h_kW=m_test["mae"],
        test_rmse_24h_kW=m_test["rmse"],
        test_mape_24h_pct=m_test["mape"],
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        run_id=run_dir.name,
    )
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(RUNS_DIR / "latest.json", "w") as f:
        json.dump({"latest_run": run_dir.name}, f, indent=2)

    print(f"\nUlozeno do {run_dir}:\n  model.json\n  best_hyperparams.json"
          f"\n  metadata.json\n  train.log")
    print(f"runs/latest.json aktualizovan na '{run_dir.name}'")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
