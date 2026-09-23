"""
Minimalni priklad klienta pro prediction_final/api/api.py - ukazuje presny
tvar pozadavku/odpovedi (viz take priklad primo v api.py docstringu). Nacte
poslednich 7 dni realnych dat z projektu (jen pro demonstraci - ostry klient
na strane meridla by posilal svuj vlastni buffer mereni) a zavola bezici API.

Predpoklada bezici server:
    python3 -m uvicorn api:app --app-dir prediction_final/api --port 8000

Pouziti:
    python3 prediction_final/api/api_client_example.py
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

API_URL = "http://127.0.0.1:8000/predict"


def main():
    df = common.load_data()
    history = df.iloc[-700:-96]  # ~7 dni, s rezervou nechano 96 realnych radku na konci pro srovnani

    readings = [
        {"time": t.tz_localize("UTC").isoformat(), "value_kW": float(v)}
        for t, v in zip(history["time"], history[common.TARGET_COL])
    ]

    print(f"Posilam {len(readings)} zaznamu ({readings[0]['time']} .. {readings[-1]['time']})")
    r = requests.post(API_URL, json={"readings": readings})
    r.raise_for_status()
    body = r.json()

    print(f"\nModel: {body['model_run_id']}  (dokumentovana MAE={body['documented_test_mae_kW']} kW,"
          f" MAPE={body['documented_test_mape_pct']}%)")
    print(f"Doplneno mezer: {body['gaps_filled_steps']} kroku\n")
    print("Prvnich 5 kroku predikce:")
    for p in body["predictions"][:5]:
        print(f"  {p['time']}  {p['value_kW']:.2f} kW")
    print(f"  ... celkem {len(body['predictions'])} kroku (24h)")


if __name__ == "__main__":
    main()
