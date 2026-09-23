"""
Sdilene jadro pro api.py (a test_api.py) v teto slozce. Toto je VLASTNI
KOPIE `prediction_final/common.py` (ne import z rodicovske slozky) - slozka
`prediction_final/api/` je zamerne izolovana, da se zkopirovat/nasadit
samostatne, bez zavislosti na zbytku `prediction_final/` (krome samotnych
natrenovanych modelu v `../runs/`, ktere vyrabi `train.py`).

Feature-set a metodologie: viz prediction/EXPERIMENT_LOG.md "Kolo 3".
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR.parent / "runs"  # natrenovane modely zustavaji spolecne v prediction_final/runs/


def latest_run_dir() -> Path:
    """Slozka nejnovejsiho tréninkoveho behu (prediction_final/runs/<run_id>/),
    dle ukazatele runs/latest.json, ktery aktualizuje train.py po kazdem behu."""
    with open(RUNS_DIR / "latest.json") as f:
        run_id = json.load(f)["latest_run"]
    return RUNS_DIR / run_id

SEED = 42

DATA_PATH = BASE_DIR.parent.parent / "data_processed" / "main_archive_15min.csv"

TARGET_COL = "Avg.3P[kW]"
RESAMPLE_MIN = 15

LAG_STEPS = [1, 2, 4, 8, 12, 24, 48, 96]  # 15,30,60,120,180,360,720,1440 min
ROLL_WINDOWS = [4, 16, 32]  # 1h, 4h, 8h

TIME_COLS = ["hour", "minute", "dayofweek", "day_sin", "day_cos",
             "month_sin", "month_cos", "year_sin", "year_cos", "is_weekend", "is_break"]

# Uzivatel oznacil data pred timto datem za mereni s chybou / hodne chybejicimi daty.
MODEL_START = pd.Timestamp("2025-01-01 00:00:00")
MAX_GAP_FILL_STEPS = 24  # interpolovat mezery do 6h

TEST_DAYS = 30
VAL_DAYS = 14
RESET_24H = 96


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


def czech_public_holidays(years) -> set:
    """Presny kalendar ceskych statnich svatku (fixni data + Velikonocni
    pondeli, dopocitane pres dateutil.easter - ne odhad z pameti)."""
    from dateutil.easter import easter
    fixed = [(1, 1), (5, 1), (5, 8), (7, 5), (7, 6), (9, 28), (10, 28), (11, 17),
             (12, 24), (12, 25), (12, 26)]
    days = set()
    for y in years:
        for mo, d in fixed:
            days.add(pd.Timestamp(y, mo, d).date())
        days.add((pd.Timestamp(easter(y)) + pd.Timedelta(days=1)).date())  # Velikonocni pondeli
    return days


def is_break_or_holiday(ts: pd.Series) -> pd.Series:
    """Budova je pracoviste VUT - spotreba se ridi hlavnim letnim provozem/
    prazdninami (cervenec+srpen) A statnimi svatky (presny cesky kalendar),
    ne jen dnem v tydnu. Viz prediction/EXPERIMENT_LOG.md "Kolo 3"."""
    years = range(ts.dt.year.min(), ts.dt.year.max() + 1)
    holidays = czech_public_holidays(years)
    is_summer = ts.dt.month.isin([7, 8])
    is_holiday = ts.dt.date.isin(holidays)
    return (is_summer | is_holiday).astype(np.int16)


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

    time_fut = df["time"] + pd.to_timedelta(RESAMPLE_MIN, unit="min")
    tf = time_feats(time_fut)
    for c in tf.columns:
        df[c] = tf[c]
    df["is_break"] = is_break_or_holiday(time_fut)

    df["TARGET"] = df[TARGET_COL].shift(-1)

    feature_cols = TIME_COLS + lag_cols
    df = df.dropna(subset=["TARGET"] + feature_cols).reset_index(drop=True)
    return FeatureSet(df=df, feature_cols=feature_cols, lag_map=lag_map)


def split_train_val_test(df: pd.DataFrame):
    test_start = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    train = df[df["time"] < val_start].reset_index(drop=True)
    val = df[(df["time"] >= val_start) & (df["time"] < test_start)].reset_index(drop=True)
    test = df[df["time"] >= test_start].reset_index(drop=True)
    return train, val, test


def autoregressive_predict(model, fs: FeatureSet, n_test: int, reset_steps=RESET_24H, t0: int | None = None):
    """24h-rekurzivni (autoregresivni) predikce s periodickym resyncem na realna
    data kazdych `reset_steps` kroku - odpovida produkcnimu nasazeni. `t0`
    (index v fs.df, odkud rekurze zacina) lze zadat rucne, jinak = konec dat
    minus n_test (klasicky "test" mod)."""
    df = fs.df
    feature_cols = fs.feature_cols
    lag_map = fs.lag_map
    n_total = len(df)
    if t0 is None:
        t0 = n_total - n_test

    time_arr = df[TIME_COLS].to_numpy(np.float32)
    target_real = df[TARGET_COL].to_numpy(np.float64)
    # +1: zahrnout i target_real[t0] samotny ("ted", posledni realne znama
    # hodnota, ze ktere rekurze vychazi) - bez ni by roll/delta featury pro
    # prvni krok cyklu a lag featury pro druhy krok cyklu chybne pouzivaly
    # vlastni predikci misto teto realne hodnoty.
    target_buffer = list(target_real[:t0 + 1])

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
        for k, c in enumerate(TIME_COLS):
            row_buf[0, col_idx[c]] = time_arr[global_idx, k]

        pred = float(np.clip(model.predict(row_buf)[0], -1e6, 1e6))
        predictions[i] = pred
        steps_since_reset += 1
        target_buffer.append(pred)

    return predictions


class SeasonalNaiveModel:
    """Predikce = hodnota pred 24h (`lag_1day`). Pouzito jako druha slozka
    `blend_autoregressive_predict` a pro alpha-search na validaci v train.py."""

    def __init__(self, feature_cols: list):
        self.lag1day_idx = feature_cols.index("lag_1day")

    def predict(self, X):
        return X[:, self.lag1day_idx]


def blend_autoregressive_predict(xgb_model, fs: FeatureSet, n_test: int, alpha: float,
                                  reset_steps=RESET_24H, t0: int | None = None):
    """alpha*XGBoost + (1-alpha)*SeasonalNaive24h - shrinkage smerem ke
    stabilni sezonni kotve snizuje rozptyl dlouho-horizontove rekurzivni
    predikce (viz prediction/EXPERIMENT_LOG.md "Kolo 3").

    DULEZITE: kazdy model bezi jako VLASTNI, NEZAVISLA 24h rekurze (vlastni
    buffer) - teprve na konci se jejich vystupni pole zprumeruji. Blendovani
    UVNITR jedne sdilene rekurze (tj. pouzit prumerovanou hodnotu jako
    zpetnovazebni vstup do bufferu) NENI totez a vychazi znatelne hur, protoze
    XGBoost pak vidi ve svych lag featurach smes vlastni a sezonni predikce
    misto vlastni ciste trajektorie, na kterou byl natrenovan."""
    preds_xgb = autoregressive_predict(xgb_model, fs, n_test, reset_steps, t0)
    seasonal_model = SeasonalNaiveModel(fs.feature_cols)
    preds_seasonal = autoregressive_predict(seasonal_model, fs, n_test, reset_steps, t0)
    return alpha * preds_xgb + (1 - alpha) * preds_seasonal


def evaluate(y_true, y_pred, mape_abs_min: float = 5.0) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > mape_abs_min)
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0 if mask.sum() else np.nan
    return dict(mae=round(float(mae), 3), rmse=round(rmse, 3), mape=round(float(mape), 3))
