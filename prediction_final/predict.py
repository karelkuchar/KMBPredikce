"""
Testovaci skript: nacte model ulozeny pomoci train.py a vypise 24h (96 kroku)
rekurzivni predikci pro prvni cyklus testovaciho okna - spolu se skutecnymi
hodnotami, aby slo vizualne overit, ze model dava rozumne vysledky.

Pouziti:
    python3 prediction_final/predict.py
"""

import json

import pandas as pd
from xgboost import XGBRegressor

import common


def main():
    run_dir = common.latest_run_dir()

    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(run_dir / "model.json"))

    with open(run_dir / "best_hyperparams.json") as f:
        hp = json.load(f)
    print(f"Beh: {run_dir.name}  (natrenovan {meta['trained_at']}, dokumentovana test MAE={meta['test_mae_24h_kW']} kW)")
    print(f"Hyperparametry: {hp}\n")

    common.TEST_DAYS = meta.get("test_days", common.TEST_DAYS)  # stejne okno, jake pouzil train.py pro tento beh

    df = common.load_data()
    fs = common.build_features(df)
    fs.feature_cols = meta["feature_cols"]
    fs.lag_map = meta["lag_map"]

    train, val, test = common.split_train_val_test(fs.df)
    t0 = len(fs.df) - len(test)

    preds = common.blend_autoregressive_predict(xgb_model, fs, n_test=96, alpha=meta["blend_alpha"],
                                                  reset_steps=None, t0=t0)
    times = fs.df["time"].iloc[t0:t0 + 96].to_numpy()
    y_true = fs.df["TARGET"].iloc[t0:t0 + 96].to_numpy()

    out = pd.DataFrame(dict(
        horizon_step=range(1, 97),
        cas=times,
        predikce_kW=preds.round(2),
        skutecnost_kW=y_true.round(2),
        chyba_kW=(preds - y_true).round(2),
    ))
    print(f"24h predikce od {times[0]} (prvni cyklus testovaciho okna):\n")
    print(out.to_string(index=False))

    mae = out["chyba_kW"].abs().mean()
    print(f"\nMAE tohoto 96-krokoveho cyklu: {mae:.2f} kW")


if __name__ == "__main__":
    main()
