"""
Sdilene jadro pro autoregresivni predikci Avg.3P[kW] (Sum_P_kW).

Vychazi z puvodniho uzivatelskeho skriptu (LightGBM, jen Sum_P_kW), rozsireno o:
  - volitelne exogenni featury (zmrazene pri rekurzi na posledni znamou hodnotu)
  - vice modelu (LightGBM / CatBoost / XGBoost)
  - rozpad chyby per krok horizontu (1..96) v ramci kazdeho rekurzivniho cyklu
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "main_archive_15min.csv"

RESAMPLE_MIN = 15
TARGET_COL = "Avg.3P[kW]"
HORIZON_STEPS = 1
HORIZON_MIN = HORIZON_STEPS * RESAMPLE_MIN

LAG_STEPS = [1, 2, 4, 8, 12, 24, 48, 96]         # 15,30,60,120,180,360,720,1440 min
ROLL_WINDOWS = [4, 16, 32]                        # 1h, 4h, 8h

EXOG_COLS = [
    "Avg.Q3[kvar]",
    "Avg.THDI1[%]", "Avg.THDI2[%]", "Avg.THDI3[%]",
    "Min.P1[kW]", "Max.P1[kW]",
    "Min.P3[kW]", "Max.P3[kW]",
]

# Kolo 2: rozsireni exogennich kanalu o napeti/proud/ucinik/THDU - test, jestli
# "rozsireni rozsireni" jeste vic pomaha (dalsi podklad pro prinos analyzatoru).
EXOG_COLS_PLUS = EXOG_COLS + [
    "Avg.U1[V]", "Avg.U2[V]", "Avg.U3[V]",
    "Avg.I1[A]", "Avg.I2[A]", "Avg.I3[A]",
    "3PF[]",
    "Avg.THDU1[%]", "Avg.THDU2[%]", "Avg.THDU3[%]",
]

# Zacatek modelovaneho useku. Puvodne 2024-12-02 (konec instalacni mezery),
# posunuto na 2025-01-01 v kole 2 - uzivatel oznacil obdobi 2024-12 za mereni
# s chybou / hodne chybejicimi daty (i kdyz to v samotnem main_archive_15min.csv
# NaN-mezery/gap analyza neukazala - domenova znalost mimo dataset).
MODEL_START = pd.Timestamp("2025-01-01 00:00:00")
MAX_GAP_FILL_STEPS = 24  # interpolovat mezery do 6h, vetsi by AR bufferu nedavaly smysl

MAPE_ABS_MIN = 5.0

TEST_DAYS = 30   # kolo 1 (srovnani_1): kratsi test okno kvuli rychlosti rekurzivniho behu
VAL_DAYS = 14

RESET_24H = 96
RESET_7D = 96 * 7


# ============================================================
# Nacteni + doplneni dat
# ============================================================
def load_base_df() -> pd.DataFrame:
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


TIME_COLS_BASE = ["hour", "minute", "dayofweek", "day_sin", "day_cos",
                   "month_sin", "month_cos", "year_sin", "year_cos", "is_weekend"]


@dataclass
class FeatureSet:
    name: str
    df: pd.DataFrame
    feature_cols: list
    lag_map: dict            # AR lag featury -> pocet 15min kroku (napr. lag_Sum_15m -> 1)
    exog_cols: list = field(default_factory=list)  # zmrazene exogenni featury pri rekurzi
    time_cols: list = field(default_factory=lambda: list(TIME_COLS_BASE))  # kalendarni featury (viz time_feats)


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
    prazdninami (cervenec+srpen, dle uzivatele) A statnimi svatky (presny
    cesky kalendar), ne jen dnem v tydnu. Pouzito v mode='univariate_semester'
    (kolo 3)."""
    years = range(ts.dt.year.min(), ts.dt.year.max() + 1)
    holidays = czech_public_holidays(years)
    is_summer = ts.dt.month.isin([7, 8])
    is_holiday = ts.dt.date.isin(holidays)
    return (is_summer | is_holiday).astype(np.int16)


def build_features(mode: str = "univariate") -> FeatureSet:
    """mode: 'univariate' (jen Sum_P_kW), 'univariate_v2' (+ lag_2week/lag_3week
    pro silnejsi tydenni signal), 'multivariate' (+ exog. featury), nebo
    'multivariate_plus' (+ napeti/proud/ucinik/THDU navic)."""
    df = load_base_df()

    lag_cols = []
    lag_map = {}
    for s in LAG_STEPS:
        label = s * RESAMPLE_MIN
        col = f"lag_Sum_{label}m"
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

    if mode == "univariate_v2":
        # Kolo 3: silnejsi tydenni signal - dalsi tydny zpet, aby model mel
        # k dispozici vic nez 1 bod pro odhad tydenni sezonnosti (napr. odliseni
        # normalniho tydne od tydne s vyjimkou). Bezny dalsi lag, zadna zmena
        # v rekurzivni logice potreba (funguje pres uz existujici lag_map mechanismus).
        df["lag_2week"] = df[TARGET_COL].shift(96 * 14)
        df["lag_3week"] = df[TARGET_COL].shift(96 * 21)
        lag_cols += ["lag_2week", "lag_3week"]
        lag_map["lag_2week"] = 96 * 14
        lag_map["lag_3week"] = 96 * 21

    df["time_fut"] = df["time"] + pd.to_timedelta(HORIZON_MIN, unit="min")
    tf = time_feats(df["time_fut"])
    for c in tf.columns:
        df[c] = tf[c]
    time_cols = list(tf.columns)

    if mode == "univariate_semester":
        # Kolo 3: budova je univerzitni pracoviste (VUT) - spotreba se ridi
        # letnim provozem/prazdninami a statnimi svatky, ne jen dnem v tydnu.
        # Viz is_break_or_holiday().
        df["is_break"] = is_break_or_holiday(df["time_fut"])
        time_cols = time_cols + ["is_break"]

    target_col = f"{TARGET_COL}_tplus{HORIZON_MIN}min"
    df[target_col] = df[TARGET_COL].shift(-HORIZON_STEPS)

    exog_cols = []
    if mode == "multivariate":
        exog_cols = list(EXOG_COLS)
    elif mode == "multivariate_plus":
        exog_cols = list(EXOG_COLS_PLUS)

    feature_cols = time_cols + lag_cols + exog_cols
    needed = [target_col] + feature_cols
    df = df.dropna(subset=needed).reset_index(drop=True)

    df = df.rename(columns={target_col: "TARGET"})
    return FeatureSet(name=mode, df=df, feature_cols=feature_cols, lag_map=lag_map,
                       exog_cols=exog_cols, time_cols=time_cols)


# ============================================================
# Split
# ============================================================
def split_train_val_test(df: pd.DataFrame):
    test_start = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    train = df[df["time"] < val_start].reset_index(drop=True)
    val = df[(df["time"] >= val_start) & (df["time"] < test_start)].reset_index(drop=True)
    test = df[df["time"] >= test_start].reset_index(drop=True)
    return train, val, test


# ============================================================
# Modely
# ============================================================
def train_lightgbm(train, val, feature_cols):
    import lightgbm as lgb
    dtrain = lgb.Dataset(train[feature_cols].to_numpy(np.float32), label=train["TARGET"].to_numpy(np.float32))
    dval = lgb.Dataset(val[feature_cols].to_numpy(np.float32), label=val["TARGET"].to_numpy(np.float32), reference=dtrain)
    params = dict(objective="regression", metric="mae", learning_rate=0.05, num_leaves=63,
                  min_child_samples=30, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, seed=SEED, verbosity=-1)
    bst = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    return lambda X: bst.predict(X, num_iteration=bst.best_iteration)


def train_catboost(train, val, feature_cols, params: dict | None = None):
    from catboost import CatBoostRegressor, Pool
    p = dict(iterations=3000, learning_rate=0.05, depth=8, loss_function="MAE",
             random_seed=SEED, early_stopping_rounds=150, verbose=False)
    if params:
        p.update(params)
    model = CatBoostRegressor(**p)
    model.fit(Pool(train[feature_cols], train["TARGET"]), eval_set=Pool(val[feature_cols], val["TARGET"]))
    return lambda X: model.predict(X)


def train_xgboost(train, val, feature_cols, params: dict | None = None):
    # .to_numpy(): xgboost rejects feature names containing "[" "]" "<", which
    # our raw column names (e.g. "Avg.Q3[kvar]") do - use plain arrays instead.
    from xgboost import XGBRegressor
    p = dict(n_estimators=3000, learning_rate=0.05, max_depth=8,
             subsample=0.8, colsample_bytree=0.9, objective="reg:absoluteerror",
             random_state=SEED, early_stopping_rounds=150, eval_metric="mae")
    if params:
        p.update(params)
    model = XGBRegressor(**p)
    model.fit(train[feature_cols].to_numpy(np.float32), train["TARGET"].to_numpy(np.float32),
              eval_set=[(val[feature_cols].to_numpy(np.float32), val["TARGET"].to_numpy(np.float32))],
              verbose=False)
    return lambda X: model.predict(X)


def _index_predict_fn(feature_cols, col_name):
    idx = feature_cols.index(col_name)
    return lambda X: X[:, idx]


def train_persistence(train, val, feature_cols):
    # predikce = posledni znama hodnota (lag 15 min)
    return _index_predict_fn(feature_cols, "lag_Sum_15m")


def train_seasonal_naive(train, val, feature_cols):
    # predikce = hodnota ve stejnem slotu pred 24h
    return _index_predict_fn(feature_cols, "lag_1day")


MODEL_TRAINERS = {
    "LightGBM": train_lightgbm,
    "CatBoost": train_catboost,
    "XGBoost": train_xgboost,
}

BASELINE_TRAINERS = {
    "Persistence": train_persistence,
    "SeasonalNaive24h": train_seasonal_naive,
}


# ============================================================
# Rychla vektorizovana 1-krokova predikce (bez rekurze, referencni horni mez)
# ============================================================
def one_step_predict(predict_fn, test: pd.DataFrame, feature_cols: list) -> np.ndarray:
    X = test[feature_cols].to_numpy(np.float32)
    return np.clip(np.asarray(predict_fn(X), dtype=np.float32), -1e6, 1e6)


# ============================================================
# Rekurzivni (autoregresivni) predikce
# ============================================================
def autoregressive_predict(predict_fn, fs: FeatureSet, n_test: int, reset_steps):
    """
    predict_fn: X (2D np.float32, 1 radek) -> y (skalar nebo 1-prvkove pole)
    fs: FeatureSet (df obsahuje sloupec TARGET_COL = realna hodnota Avg.3P[kW] v case t,
        pouzita jako zdroj pro AR buffer a pro zmrazene exogenni featury)
    reset_steps: None (bez resetu), nebo pocet kroku mezi resynchronizacemi na realna data

    Vraci: predictions (np.float32, delka n_test), horizon_step (np.int32, pozice v ramci
        aktualniho rekurzivniho cyklu, 1-indexovana - pro rozpad chyby po horizontech).
    """
    df = fs.df
    feature_cols = fs.feature_cols
    lag_map = fs.lag_map
    n_total = len(df)
    t0 = n_total - n_test

    time_cols = fs.time_cols
    time_arr = df[time_cols].to_numpy(np.float32)          # precompute, staticke featury
    exog_arr = df[fs.exog_cols].to_numpy(np.float64) if fs.exog_cols else None

    target_real = df[TARGET_COL].to_numpy(np.float64)
    # target_buffer[k] == target_real[k] (synced) nebo predikce, ktera ji
    # aproximuje. Zahrnuje i target_real[t0] - "ted", posledni realne
    # znama hodnota, ze ktere rekurze vychazi (bez ni by roll/delta featury
    # pro prvni krok cyklu i lag featury pro druhy krok cyklu chybne
    # pouzivaly vlastni predikci misto teto realne hodnoty).
    target_buffer = list(target_real[:t0 + 1])

    exog_frozen = exog_arr[t0].copy() if exog_arr is not None else None

    predictions = np.full(n_test, np.nan, dtype=np.float32)
    horizon_step = np.zeros(n_test, dtype=np.int32)
    steps_since_reset = 0

    # index kazdeho feature sloupce v X radku - pripravit mapovani jednou
    col_idx = {c: k for k, c in enumerate(feature_cols)}
    row_buf = np.zeros((1, len(feature_cols)), dtype=np.float32)

    for i in range(n_test):
        global_idx = t0 + i

        if reset_steps is not None and steps_since_reset >= reset_steps:
            # posledni `steps_since_reset` pozice bufferu (global_idx-steps_since_reset+1 .. global_idx)
            # drzely predikce z prave dokonceneho cyklu - prepsat je realnymi hodnotami.
            resync_from = global_idx - steps_since_reset + 1
            for k in range(resync_from, global_idx + 1):
                if 0 <= k < len(target_buffer):
                    target_buffer[k] = float(target_real[k])
            if exog_arr is not None:
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

        for k, c in enumerate(fs.exog_cols):
            row_buf[0, col_idx[c]] = exog_frozen[k]

        for k, c in enumerate(time_cols):
            row_buf[0, col_idx[c]] = time_arr[global_idx, k]

        # .copy(): CatBoost's predict() marks the input array read-only as a
        # side effect, which would break reuse of row_buf on the next iteration.
        pred = float(np.clip(predict_fn(row_buf.copy())[0], -1e6, 1e6))

        predictions[i] = pred
        steps_since_reset += 1
        horizon_step[i] = steps_since_reset
        target_buffer.append(pred)

    return predictions, horizon_step


# ============================================================
# Staggered (rolling-start) rekurzivni evaluace - oprava fazoveho zkresleni
# ============================================================
# autoregressive_predict() vzdy resynchronizuje po presne `reset_steps`
# krocich od pevneho t0 -> horizont krok je zavisly na hodine dne (napr.
# horizont krok 48 = vzdy stejna hodina +12h od resetu napric cyklama).
# Tato varianta spousti nezavisly `horizon`-krokovy rekurzivni beh z mnoha
# ruznych startovnich bodu (rozprostrenych po testovacim okne, krok `stride`),
# takze pro dany horizont krok h jde napric behy o ruzne hodiny/dny -> per-h
# MAE uz odrazi cistou kumulaci chyby rekurzi, ne obtiznost konkretni hodiny.
MAX_LOOKBACK = max([96 * 7, 96] + LAG_STEPS + ROLL_WINDOWS) + 4  # >= nejdelsi pouzity lag (lag_1week) - vychozi feature-sety


def _predict_step(predict_fn, feature_cols, col_idx, lag_map, exog_cols, time_cols, time_arr,
                   row_buf, target_buffer, exog_frozen, global_idx):
    buf_len = len(target_buffer)
    for col, lag_s in lag_map.items():
        buf_idx = buf_len - 1 - lag_s
        row_buf[0, col_idx[col]] = target_buffer[buf_idx] if buf_idx >= 0 else target_buffer[0]

    for w in ROLL_WINDOWS:
        arr = target_buffer[-w:]
        row_buf[0, col_idx[f"roll_mean_{w}s"]] = np.mean(arr) if arr else 0.0
        row_buf[0, col_idx[f"roll_std_{w}s"]] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0

    row_buf[0, col_idx["delta_1s"]] = target_buffer[-1] - target_buffer[-2] if len(target_buffer) >= 2 else 0.0

    for k, c in enumerate(exog_cols):
        row_buf[0, col_idx[c]] = exog_frozen[k]

    for k, c in enumerate(time_cols):
        row_buf[0, col_idx[c]] = time_arr[global_idx, k]

    return float(np.clip(predict_fn(row_buf.copy())[0], -1e6, 1e6))


def staggered_recursive_predict(predict_fn, fs: FeatureSet, start_indices, horizon: int = 96):
    """
    Nezavisly `horizon`-krokovy rekurzivni beh z kazdeho indexu v `start_indices`
    (0-based index do fs.df; prvni predikovany krok = df.iloc[start]).

    Vraci DataFrame se sloupci: start_idx, horizon_step (1..horizon), global_idx,
    y_true, y_pred.
    """
    df = fs.df
    feature_cols = fs.feature_cols
    lag_map = fs.lag_map
    n_total = len(df)

    time_cols = fs.time_cols
    time_arr = df[time_cols].to_numpy(np.float32)
    exog_arr = df[fs.exog_cols].to_numpy(np.float64) if fs.exog_cols else None
    target_real = df[TARGET_COL].to_numpy(np.float64)

    col_idx = {c: k for k, c in enumerate(feature_cols)}
    row_buf = np.zeros((1, len(feature_cols)), dtype=np.float32)
    # dynamicke, ne pevny modulovy konstant: nektere mody (napr. univariate_v2)
    # pouzivaji delsi lagy (lag_3week = 2016 kroku) nez vychozi MAX_LOOKBACK pokryva.
    max_lookback = max([MAX_LOOKBACK] + list(lag_map.values())) + 4

    out_start, out_hstep, out_gidx, out_pred = [], [], [], []

    for start in start_indices:
        if start < 1 or start + horizon >= n_total:  # >=: y_true potrebuje jeste index (start+horizon)
            continue
        lookback_from = max(0, start - max_lookback)
        # +1: zahrnout i target_real[start] samotny ("ted", viz autoregressive_predict)
        target_buffer = list(target_real[lookback_from:start + 1])
        exog_frozen = exog_arr[start].copy() if exog_arr is not None else None

        for step in range(1, horizon + 1):
            global_idx = start + step - 1
            pred = _predict_step(predict_fn, feature_cols, col_idx, lag_map, fs.exog_cols, time_cols,
                                  time_arr, row_buf, target_buffer, exog_frozen, global_idx)
            out_start.append(start)
            out_hstep.append(step)
            out_gidx.append(global_idx)
            out_pred.append(pred)
            target_buffer.append(pred)

    y_pred = np.array(out_pred, dtype=np.float32)
    # predictions[k] aproximuje TARGET pri global_idx, tj. target_real[global_idx+1]
    # (o krok dal, stejna konvence jako fs.df["TARGET"] = target_real.shift(-1)).
    y_true = target_real[np.array(out_gidx) + 1].astype(np.float32) if out_gidx else np.array([], dtype=np.float32)
    return pd.DataFrame(dict(start_idx=out_start, horizon_step=out_hstep, global_idx=out_gidx,
                              y_true=y_true, y_pred=y_pred))


# ============================================================
# Metriky
# ============================================================
def mape_masked(y_true, y_pred, abs_min=MAPE_ABS_MIN):
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    mask = finite & (np.abs(y_true) > abs_min)
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = mape_masked(y_true, y_pred)
    return dict(mae=round(float(mae), 5), rmse=round(rmse, 5), mape=round(float(mape), 3))
