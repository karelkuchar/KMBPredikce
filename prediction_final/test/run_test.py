"""
Spusti finalni model (nejnovejsi beh z prediction_final/runs/, dle
runs/latest.json) na celem testovacim okne (24h rekurze, reset kazdych 96
kroku - stejny rezim jako oficialni metrika v train.py) a ulozi vysledek na
DVE MISTA - do `test/` (vzdy "posledni vysledek") a do prislusne
`runs/<run_id>/` (aby vysledky zustaly svazane s konkretnim behem, stejne
jako model/metadata - historie se neprepisuje):

    test_results.csv           - cas, skutecnost, predikce, chyba, horizon_step
    test_plot.png              - graf skutecnost vs. predikce pres cele test okno
    test_metrics_breakdown.csv - MAE/MAPE rozepsane po hodine dne, dni a mesici

Pouziti:
    python3 prediction_final/test/run_test.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import common  # noqa: E402

COLOR_REAL = "#2a78d6"
COLOR_PRED = "#eb6834"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"


def main():
    run_dir = common.latest_run_dir()

    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(run_dir / "model.json"))

    common.TEST_DAYS = meta.get("test_days", common.TEST_DAYS)  # stejne okno, jake pouzil train.py pro tento beh

    df = common.load_data()
    fs = common.build_features(df)
    fs.feature_cols = meta["feature_cols"]
    fs.lag_map = meta["lag_map"]
    alpha = meta["blend_alpha"]
    seasonal_model = common.SeasonalNaiveModel(fs.feature_cols)

    print(f"Beh: {run_dir.name}  (blend alpha={alpha})")
    train, val, test = common.split_train_val_test(fs.df)
    t0 = len(fs.df) - len(test)

    # blend = dve NEZAVISLE rekurze (XGBoost, SeasonalNaive24h), zprumerovane
    # az na konci - viz common.blend_autoregressive_predict pro vysvetleni,
    # proc blendovani uvnitr jedne sdilene rekurze neni totez a vychazi hur.
    preds_xgb, horizon_step = _predict_with_horizon(xgb_model, fs, len(test), common.RESET_24H, t0)
    preds_seasonal, _ = _predict_with_horizon(seasonal_model, fs, len(test), common.RESET_24H, t0)
    preds = alpha * preds_xgb + (1 - alpha) * preds_seasonal
    times = fs.df["time"].iloc[t0:].to_numpy()
    y_true = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)

    out = pd.DataFrame(dict(
        cas=times,
        horizon_step=horizon_step,
        skutecnost_kW=y_true.round(3),
        predikce_kW=preds.round(3),
        chyba_kW=(preds - y_true).round(3),
    ))
    for d in (OUT_DIR, run_dir):
        out.to_csv(d / "test_results.csv", index=False)

    m = common.evaluate(y_true, preds)
    print(f"Test okno: {times[0]} -> {times[-1]}  ({len(out)} kroku)")
    print(f"MAE={m['mae']} kW  RMSE={m['rmse']} kW  MAPE={m['mape']}%")

    plot_results(out, m, run_dir)
    breakdown = save_metrics_breakdown(out, run_dir)
    print(f"\nUlozeno do {OUT_DIR} a {run_dir}:\n  test_results.csv\n  test_plot.png\n  test_metrics_breakdown.csv")
    print(f"\nRozpad metrik (prvnich 5 radku z {len(breakdown)}):")
    print(breakdown.head().to_string(index=False))


def save_metrics_breakdown(out: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """MAE/MAPE rozepsane po hodine dne (0-23), kalendarnim dni a mesici -
    umoznuje videt, jestli chyba neni rovnomerne rozlozena (napr. hor v
    urcitou hodinu nebo mesic) misto jen jednoho celkoveho cisla."""
    df = out.copy()
    df["cas"] = pd.to_datetime(df["cas"])

    def agg(group_col, label):
        rows = []
        for key, g in df.groupby(group_col):
            m = common.evaluate(g["skutecnost_kW"].to_numpy(), g["predikce_kW"].to_numpy())
            rows.append(dict(granularita=label, obdobi=str(key), n=len(g), mae_kW=m["mae"],
                              rmse_kW=m["rmse"], mape_pct=m["mape"]))
        return rows

    rows = []
    rows += agg(df["cas"].dt.hour, "hodina_dne")
    rows += agg(df["cas"].dt.date, "den")
    rows += agg(df["cas"].dt.to_period("M"), "mesic")

    breakdown = pd.DataFrame(rows)
    for d in (OUT_DIR, run_dir):
        breakdown.to_csv(d / "test_metrics_breakdown.csv", index=False)
    return breakdown


def _predict_with_horizon(model, fs, n_test, reset_steps, t0):
    """Stejne jako common.autoregressive_predict, jen navic vraci horizon_step
    (pozice v ramci aktualniho 96-krokoveho cyklu) pro popisky v grafu/CSV."""
    df = fs.df
    feature_cols = fs.feature_cols
    lag_map = fs.lag_map
    time_arr = df[common.TIME_COLS].to_numpy(np.float32)
    target_real = df[common.TARGET_COL].to_numpy(np.float64)
    target_buffer = list(target_real[:t0 + 1])

    predictions = np.full(n_test, np.nan, dtype=np.float32)
    horizon_step = np.zeros(n_test, dtype=np.int32)
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
        for w in common.ROLL_WINDOWS:
            arr = target_buffer[-w:]
            row_buf[0, col_idx[f"roll_mean_{w}s"]] = np.mean(arr) if arr else 0.0
            row_buf[0, col_idx[f"roll_std_{w}s"]] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        row_buf[0, col_idx["delta_1s"]] = target_buffer[-1] - target_buffer[-2] if len(target_buffer) >= 2 else 0.0
        for k, c in enumerate(common.TIME_COLS):
            row_buf[0, col_idx[c]] = time_arr[global_idx, k]

        pred = float(np.clip(model.predict(row_buf)[0], -1e6, 1e6))
        predictions[i] = pred
        steps_since_reset += 1
        horizon_step[i] = steps_since_reset
        target_buffer.append(pred)

    return predictions, horizon_step


def plot_results(out: pd.DataFrame, metrics: dict, run_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    cas = pd.to_datetime(out["cas"])
    ax.plot(cas, out["skutecnost_kW"], color=COLOR_REAL, linewidth=1.3, label="Skutečnost")
    ax.plot(cas, out["predikce_kW"], color=COLOR_PRED, linewidth=1.3, label="Predikce")

    fig.text(0.02, 0.97, "Test: skutečnost vs. 24h rekurzivní predikce (Avg.3P[kW])",
             color=COLOR_TEXT, fontsize=13, ha="left", va="top")
    fig.text(0.02, 0.925, f"MAE={metrics['mae']} kW   RMSE={metrics['rmse']} kW   MAPE={metrics['mape']}%",
             color=COLOR_TEXT_SECONDARY, fontsize=10, ha="left", va="top")
    ax.set_ylabel("Avg.3P [kW]", color=COLOR_TEXT_SECONDARY)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d8d7d0")
    ax.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color="#e8e7e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", labelcolor=COLOR_TEXT)

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.88])  # 0.88: reservuje mesto nahore pro title + MAE radek (fig.text)
    for d in (OUT_DIR, run_dir):
        fig.savefig(d / "test_plot.png", dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
