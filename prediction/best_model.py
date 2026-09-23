"""
Samostatny, izolovany skript s viteznou konfiguraci (kolo 2):
feature-set 'multivariate' (Q3, THDI1-3, Min/Max P1/P3) + XGBoost s vyladenymi
hyperparametry. Zamerne nezavisi na prediction/pipeline.py ani na zbytku
srovnani_1/srovnani_2 - da se zkopirovat a pouzit samostatne (zaklad pro
budouci Python API/serving skript, viz UKOL.md).

Dokumentovana ocekavana presnost (test, 2025-01-01+ okno, viz
prediction/EXPERIMENT_LOG.md "Kolo 2"): 24h rekurzivni MAE ≈ 25-27 kW.

Pouziti:
    python3 prediction/best_model.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42

DATA_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "main_archive_15min.csv"

TARGET_COL = "Avg.3P[kW]"
RESAMPLE_MIN = 15
HORIZON_MIN = 15  # jeden AR krok = 15 min; 24h dopredu = 96x rekurzivni aplikace

LAG_STEPS = [1, 2, 4, 8, 12, 24, 48, 96]  # 15,30,60,120,180,360,720,1440 min
ROLL_WINDOWS = [4, 16, 32]  # 1h, 4h, 8h

EXOG_COLS = [
    "Avg.Q3[kvar]",
    "Avg.THDI1[%]", "Avg.THDI2[%]", "Avg.THDI3[%]",
    "Min.P1[kW]", "Max.P1[kW]",
    "Min.P3[kW]", "Max.P3[kW]",
]

# Uzivatel oznacil data pred timto datem za mereni s chybou / hodne chybejicimi daty.
MODEL_START = pd.Timestamp("2025-01-01 00:00:00")
MAX_GAP_FILL_STEPS = 24  # interpolovat mezery do 6h

TEST_DAYS = 30
VAL_DAYS = 14
RESET_24H = 96

# Vitezne hyperparametry z prediction/srovnani_2/tune.py (25-trial random
# search, cileno na 24h rekurzivni validacni MAE, ne na 1step).
XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, max_depth=6,
    subsample=0.7, colsample_bytree=1.0, min_child_weight=1,
    reg_alpha=0.5, reg_lambda=5.0,
    objective="reg:absoluteerror", random_state=SEED,
    early_stopping_rounds=150, eval_metric="mae",
)


# ============================================================
# Data + featury
# ============================================================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Record Time[s]"])
    df = df.rename(columns={"Record Time[s]": "time"}).set_index("time").sort_index()
    df = df[df.index >= MODEL_START]
    df = df.interpolate(limit=MAX_GAP_FILL_STEPS).bfill().ffill()
    return df.reset_index()


def time_feats(ts: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=ts.index)
    out["hour"] = ts.dt.hour.astype(np.int16)
    out["minute"] = ts.dt.minute.astype(np.int16)
    out["dayofweek"] = ts.dt.dayofweek.astype(np.int16)
    slot = (out["hour"] * 4 + out["minute"] // 15).astype(np.float32)
    out["day_sin"] = np.sin(2 * np.pi * slot / 96).astype(np.float32)
    out["day_cos"] = np.cos(2 * np.pi * slot / 96).astype(np.float32)
    m = ts.dt.month.astype(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * m / 12).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * m / 12).astype(np.float32)
    doy = ts.dt.dayofyear.astype(np.float32)
    out["year_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    out["year_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(np.int16)
    return out


TIME_COLS = ["hour", "minute", "dayofweek", "day_sin", "day_cos",
             "month_sin", "month_cos", "year_sin", "year_cos", "is_weekend"]


@dataclass
class FeatureSet:
    df: pd.DataFrame
    feature_cols: list
    lag_map: dict


def build_features(df: pd.DataFrame) -> FeatureSet:
    df = df.copy()
    lag_cols, lag_map = [], {}
    for s in LAG_STEPS:
        col = f"lag_Sum_{s * RESAMPLE_MIN}m"
        df[col] = df[TARGET_COL].shift(s)
        lag_cols.append(col)
        lag_map[col] = s

    for w in ROLL_WINDOWS:
        mcol, scol = f"roll_mean_{w}s", f"roll_std_{w}s"
        df[mcol] = df[TARGET_COL].rolling(w, min_periods=1).mean().astype(np.float32)
        df[scol] = df[TARGET_COL].rolling(w, min_periods=1).std().fillna(0).astype(np.float32)
        lag_cols += [mcol, scol]

    df["delta_1s"] = (df[TARGET_COL] - df[TARGET_COL].shift(1)).astype(np.float32)
    lag_cols.append("delta_1s")

    df["lag_1week"] = df[TARGET_COL].shift(96 * 7)
    df["lag_1day"] = df[TARGET_COL].shift(96)
    lag_cols += ["lag_1week", "lag_1day"]
    lag_map["lag_1week"] = 96 * 7
    lag_map["lag_1day"] = 96

    tf = time_feats(df["time"] + pd.to_timedelta(HORIZON_MIN, unit="min"))
    for c in tf.columns:
        df[c] = tf[c]

    df["TARGET"] = df[TARGET_COL].shift(-1)

    feature_cols = TIME_COLS + lag_cols + EXOG_COLS
    df = df.dropna(subset=["TARGET"] + feature_cols).reset_index(drop=True)
    return FeatureSet(df=df, feature_cols=feature_cols, lag_map=lag_map)


def split_train_val_test(df: pd.DataFrame):
    test_start = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    train = df[df["time"] < val_start].reset_index(drop=True)
    val = df[(df["time"] >= val_start) & (df["time"] < test_start)].reset_index(drop=True)
    test = df[df["time"] >= test_start].reset_index(drop=True)
    return train, val, test


# ============================================================
# Model
# ============================================================
def train_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[feature_cols].to_numpy(np.float32), train["TARGET"].to_numpy(np.float32),
              eval_set=[(val[feature_cols].to_numpy(np.float32), val["TARGET"].to_numpy(np.float32))],
              verbose=False)
    return model


def autoregressive_predict(model: XGBRegressor, fs: FeatureSet, n_test: int, reset_steps=RESET_24H):
    """24h-rekurzivni (autoregresivni) predikce s periodickym resyncem na realna
    data kazdych `reset_steps` kroku - odpovida produkcnimu nasazeni (viz
    prediction/pipeline.py::autoregressive_predict pro plnou dokumentaci logiky,
    zde 1:1 stejny algoritmus, jen bez zavislosti na pipeline.py)."""
    df = fs.df
    feature_cols = fs.feature_cols
    lag_map = fs.lag_map
    n_total = len(df)
    t0 = n_total - n_test

    time_arr = df[TIME_COLS].to_numpy(np.float32)
    exog_arr = df[EXOG_COLS].to_numpy(np.float64)
    target_real = df[TARGET_COL].to_numpy(np.float64)
    # +1: zahrnout i target_real[t0] samotny ("ted", posledni realne znama
    # hodnota, ze ktere rekurze vychazi) - bez ni by roll/delta featury pro
    # prvni krok cyklu a lag featury pro druhy krok cyklu chybne pouzivaly
    # vlastni predikci misto teto realne hodnoty.
    target_buffer = list(target_real[:t0 + 1])
    exog_frozen = exog_arr[t0].copy()

    predictions = np.full(n_test, np.nan, dtype=np.float32)
    steps_since_reset = 0
    col_idx = {c: k for k, c in enumerate(feature_cols)}
    row_buf = np.zeros((1, len(feature_cols)), dtype=np.float32)

    for i in range(n_test):
        global_idx = t0 + i

        if reset_steps is not None and steps_since_reset >= reset_steps:
            resync_from = global_idx - steps_since_reset + 1
            for k in range(resync_from, global_idx + 1):
                if 0 <= k < len(target_buffer):
                    target_buffer[k] = float(target_real[k])
            exog_frozen = exog_arr[global_idx].copy()
            steps_since_reset = 0

        buf_len = len(target_buffer)
        for col, lag_s in lag_map.items():
            buf_idx = buf_len - 1 - lag_s
            row_buf[0, col_idx[col]] = target_buffer[buf_idx] if buf_idx >= 0 else target_buffer[0]
        for w in ROLL_WINDOWS:
            arr = target_buffer[-w:]
            row_buf[0, col_idx[f"roll_mean_{w}s"]] = np.mean(arr) if arr else 0.0
            row_buf[0, col_idx[f"roll_std_{w}s"]] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        row_buf[0, col_idx["delta_1s"]] = target_buffer[-1] - target_buffer[-2] if len(target_buffer) >= 2 else 0.0
        for k, c in enumerate(EXOG_COLS):
            row_buf[0, col_idx[c]] = exog_frozen[k]
        for k, c in enumerate(TIME_COLS):
            row_buf[0, col_idx[c]] = time_arr[global_idx, k]

        pred = float(np.clip(model.predict(row_buf)[0], -1e6, 1e6))
        predictions[i] = pred
        steps_since_reset += 1
        target_buffer.append(pred)

    return predictions


def evaluate(y_true, y_pred, mape_abs_min: float = 5.0) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > mape_abs_min)
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0 if mask.sum() else np.nan
    return dict(mae=round(float(mae), 3), rmse=round(rmse, 3), mape=round(float(mape), 3))


# ============================================================
# CLI
# ============================================================
def main():
    df = load_data()
    fs = build_features(df)
    train, val, test = split_train_val_test(fs.df)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  features={len(fs.feature_cols)}")

    model = train_model(train, val, fs.feature_cols)
    t0 = len(fs.df) - len(test)
    y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)

    preds_24h = autoregressive_predict(model, fs, n_test=len(test), reset_steps=RESET_24H)
    m24 = evaluate(y_true, preds_24h)
    print(f"24h MAE={m24['mae']} kW  RMSE={m24['rmse']} kW  MAPE={m24['mape']}%")
    print("(ocekavano ~25-27 kW dle delky test okna, viz prediction/EXPERIMENT_LOG.md)")


if __name__ == "__main__":
    main()
