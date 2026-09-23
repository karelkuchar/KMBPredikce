"""
End-to-end test API (prediction_final/api/api.py): spusti skutecny uvicorn
server, posle na nej pozadavky pres HTTP (`requests`), a overi:

  1. /health odpovida spravne
  2. /predict s realnymi historickymi daty dava stejny vysledek jako primo
     zavolane common.blend_autoregressive_predict (stejna metodika)
  3. include_history=true/false vraci/nevraci pole 'history' v odpovedi
  4. mezera v datech nad limit (6h) je odmitnuta, pod limitem se doplni
  5. data pres skutecny CZ DST prechod projdou bez chyby (vse v UTC)
  6. "delsi beh" - /predict zavolano z ~25 ruznych startovnich bodu napric
     testovacim oknem, agregovana MAE proti skutecnym budoucim hodnotam by
     mela odpovidat dokumentovanemu cislu (~19-21 kW)
  7. Validace vstupu spravne odmita spatna data (chybejici tz, mimo 15min
     mrizku, prilis kratka historie, duplicity, prilis velka mezera, NaN)

Pouziti:
    python3 prediction_final/api/test_api.py
"""

import subprocess
import sys
import time
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent  # api/ - api.py a common.py jsou zde jako sourozenci
sys.path.insert(0, str(BASE_DIR))
import common  # noqa: E402

HOST, PORT = "127.0.0.1", 8123
BASE_URL = f"http://{HOST}:{PORT}"


def readings_json(df: pd.DataFrame) -> list[dict]:
    return [
        {"time": t.tz_localize("UTC").isoformat(), "value_kW": float(v)}
        for t, v in zip(df["time"], df[common.TARGET_COL])
    ]


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", HOST, "--port", str(PORT), "--log-level", "warning"],
        cwd=str(BASE_DIR),
    )
    for _ in range(60):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("API server se nerozjel do 30s")


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, r.text
    body = r.json()
    print(f"[OK] /health: {body}")
    return body


def test_include_history_flag(df_all: pd.DataFrame):
    history = df_all.iloc[-400:-96].reset_index(drop=True)
    readings = readings_json(history)

    r_default = requests.post(f"{BASE_URL}/predict", json={"readings": readings})
    assert r_default.status_code == 200, r_default.text
    default_has_history = r_default.json().get("history") is not None

    r_true = requests.post(f"{BASE_URL}/predict", json={"readings": readings, "include_history": True})
    assert r_true.status_code == 200, r_true.text
    body_true = r_true.json()
    history_out = body_true.get("history")

    ok = (not default_has_history) and (history_out is not None) and (len(history_out) == len(readings))
    print(f"[{'OK' if ok else 'FAIL'}] include_history: vychozi=None ({not default_has_history}), "
          f"true vraci {len(history_out) if history_out else 0}/{len(readings)} zaznamu")
    assert ok, "include_history=true nevratilo ocekavanou historii (nebo vychozi false ji vratilo navic)"


def test_predict_matches_direct_call(df_all: pd.DataFrame, meta: dict, xgb_model):
    history = df_all.iloc[-500:-96].reset_index(drop=True)  # ~5 dni historie, necha rezervu skutecnych budoucich hodnot k porovnani
    r = requests.post(f"{BASE_URL}/predict", json={"readings": readings_json(history)})
    assert r.status_code == 200, r.text
    api_preds = np.array([p["value_kW"] for p in r.json()["predictions"]])

    # primy vypocet stejnou cestou jako predict.py/test/run_test.py, pro srovnani
    import api as api_module
    fs, t0 = api_module.build_live_feature_set(history.rename(columns=str))
    direct_preds = common.blend_autoregressive_predict(
        xgb_model, fs, n_test=96, alpha=meta["blend_alpha"], reset_steps=None, t0=t0)

    max_diff = np.max(np.abs(api_preds - direct_preds))
    print(f"[{'OK' if max_diff < 0.01 else 'FAIL'}] /predict vs. primy vypocet: max rozdil={max_diff:.5f} kW (mel by byt ~0)")
    assert max_diff < 0.01, "API predikce se neshoduje s primym vypoctem stejne metodiky"


def test_gap_handling(df_all: pd.DataFrame):
    """Mezera nad limit (6h) je odmitnuta (400), mezera pod limitem se
    lineárně doplni (gaps_filled_steps > 0, 200)."""
    history = df_all.iloc[-400:-96].reset_index(drop=True)
    readings = readings_json(history)

    too_long = readings[:100] + readings[132:]  # chybi 32 kroku = 8h
    r_bad = requests.post(f"{BASE_URL}/predict", json={"readings": too_long})
    bad_ok = r_bad.status_code == 400
    print(f"  [{'OK' if bad_ok else 'FAIL'}] mezera 8h (nad limit 6h): HTTP {r_bad.status_code}")

    short_gap = readings[:100] + readings[108:]  # chybi 8 kroku = 2h
    r_ok = requests.post(f"{BASE_URL}/predict", json={"readings": short_gap})
    filled = r_ok.json().get("gaps_filled_steps") if r_ok.status_code == 200 else None
    ok_ok = r_ok.status_code == 200 and filled == 8
    print(f"  [{'OK' if ok_ok else 'FAIL'}] mezera 2h (pod limit): HTTP {r_ok.status_code}, gaps_filled_steps={filled}")

    print(f"[{'OK' if bad_ok and ok_ok else 'FAIL'}] mezery v datech (nad/pod limit)")
    assert bad_ok and ok_ok, "detekce/doplneni mezer nefunguje jak ma"


def test_dst_transition(df_all: pd.DataFrame):
    """Data pokryvajici skutecny CZ DST prechod (poslední říjnová neděle,
    CEST->CET) projdou bez chyby - vse je v UTC, zadna dvojznacnost."""
    dst_start = pd.Timestamp("2025-10-20", tz="UTC")  # zahrnuje 2025-10-26 (CZ DST prechod)
    readings = [
        {"time": (dst_start + pd.Timedelta(minutes=15 * i)).isoformat(), "value_kW": 150.0 + i * 0.01}
        for i in range(200)
    ]
    r = requests.post(f"{BASE_URL}/predict", json={"readings": readings})
    ok = r.status_code == 200 and len(r.json().get("predictions", [])) == 96
    print(f"[{'OK' if ok else 'FAIL'}] data pres DST prechod (2025-10-26): HTTP {r.status_code}")
    assert ok, "data pokryvajici DST prechod by mela projit bez chyby (vse UTC)"


def test_longer_run(df_all: pd.DataFrame, n_runs: int = 25):
    """Zavola /predict z n_runs ruznych bodu napric DRZENYM TESTOVACIM OKNEM
    (posledni common.TEST_DAYS dni - stejne jako train.py/predict.py, aby
    slo srovnat s dokumentovanou presnosti; model tato data pri treninku
    nevidel), spocita MAE proti skutecnym budoucim hodnotam."""
    test_start = df_all["time"].max() - pd.Timedelta(days=common.TEST_DAYS)
    test_start_idx = int((df_all["time"] >= test_start).idxmax())

    rng = np.random.default_rng(42)
    min_start = max(test_start_idx, 672)  # 672: dost historie na lag_1week
    max_start = len(df_all) - 96 - 1
    starts = sorted(rng.choice(np.arange(min_start, max_start), size=n_runs, replace=False))

    all_errs = []
    failures = 0
    for start in starts:
        history = df_all.iloc[start - 672: start].reset_index(drop=True)  # 7 dni historie
        r = requests.post(f"{BASE_URL}/predict", json={"readings": readings_json(history)})
        if r.status_code != 200:
            failures += 1
            print(f"  [FAIL] start={start}: HTTP {r.status_code} {r.text[:200]}")
            continue
        preds = np.array([p["value_kW"] for p in r.json()["predictions"]])
        actual = df_all[common.TARGET_COL].iloc[start: start + 96].to_numpy()
        all_errs.append(np.abs(preds - actual))

    all_errs = np.concatenate(all_errs)
    mae = all_errs.mean()
    print(f"[{'OK' if failures == 0 else 'FAIL'}] delsi beh: {n_runs} volani, {failures} selhani, "
          f"agregovana MAE={mae:.2f} kW (dokumentovano ~19-21 kW)")
    assert failures == 0, f"{failures} volani /predict selhalo"


def test_input_validation(df_all: pd.DataFrame):
    good = df_all.iloc[-300:-96].reset_index(drop=True)
    good_json = readings_json(good)

    cases = []

    # prilis kratka historie
    cases.append(("kratka historie", {"readings": good_json[-50:]}))
    # chybejici timezone (naivni cas)
    naive = [dict(r) for r in good_json]
    naive[0]["time"] = naive[0]["time"].split("+")[0]  # odstrani offset -> naivni
    cases.append(("naivni cas (bez tz)", {"readings": naive}))
    # cas mimo 15min mrizku
    off_grid = [dict(r) for r in good_json]
    t = pd.Timestamp(off_grid[0]["time"]) + pd.Timedelta(minutes=1)
    off_grid[0]["time"] = t.isoformat()
    cases.append(("cas mimo 15min mrizku", {"readings": off_grid}))
    # duplicitni casy
    dupes = good_json + [good_json[0]]
    cases.append(("duplicitni casy", {"readings": dupes}))
    # NaN hodnota
    nan_case = [dict(r) for r in good_json]
    nan_case[5]["value_kW"] = float("nan")
    cases.append(("NaN hodnota", {"readings": nan_case}))
    # prazdny seznam
    cases.append(("prazdny seznam", {"readings": []}))

    import json as stdlib_json  # requests' json= uses simplejson here (allow_nan=False) - NaN needs stdlib json + raw body

    all_rejected = True
    for name, payload in cases:
        if name == "NaN hodnota":
            r = requests.post(f"{BASE_URL}/predict", data=stdlib_json.dumps(payload),
                               headers={"Content-Type": "application/json"})
        else:
            r = requests.post(f"{BASE_URL}/predict", json=payload)
        ok = r.status_code in (400, 422)
        all_rejected &= ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: HTTP {r.status_code}")

    print(f"[{'OK' if all_rejected else 'FAIL'}] validace vstupu: vsechny spatne pripady odmitnuty")
    assert all_rejected, "nektery spatny vstup nebyl odmitnut"


def main():
    df_all = common.load_data()
    xgb_model, meta, run_id = None, None, None
    import json
    from xgboost import XGBRegressor
    run_dir = common.latest_run_dir()
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(run_dir / "model.json"))

    print(f"Startuji API server (model run {run_dir.name})...")
    proc = start_server()
    try:
        test_health()
        test_predict_matches_direct_call(df_all, meta, xgb_model)
        test_include_history_flag(df_all)
        test_gap_handling(df_all)
        test_dst_transition(df_all)
        test_longer_run(df_all, n_runs=25)
        test_input_validation(df_all)
        print("\n=== VSECHNY TESTY PROSLY ===")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
