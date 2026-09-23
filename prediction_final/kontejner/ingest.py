"""
Jediny kontejner, ktery pise do InfluxDB. NEcte zadna data sam - nema k nim
pristup (bezi mimo, prenosny) - misto toho ho NEKDO ZVENKU VOLA s novymi
merenimi (POST /ingest). Pri kazdem prijatem volani:

  1. Ulozi prijata mereni do Influxu (measurement "skutecnost").
  2. Dotaze se Influxu na poslednich HISTORY_HOURS hodin "skutecnost" pro
     dany mereni (Influx = spolecna pamet, nezavisla na tom, jestli volajici
     posle 1 novy bod nebo cely 48h balik najednou).
  3. Pokud je historie dost dlouha (>= MIN_HISTORY_STEPS), zavola predikcni
     API (bezstavove, samo nic neuklada) a vyslednych 96 bodu ulozi do
     Influxu (measurement "predikce"), otagovane "predicted_at" (kdy tato
     konkretni predikce vznikla - aby se pri dalsim volani nepremazaly
     stare predikce na stejnych budoucich casech).

Pouziti (v kontejneru, viz docker-compose.yml):
    uvicorn ingest:app --host 0.0.0.0 --port 8001

Priklad volani zvenku:
    curl -X POST http://localhost:8001/ingest -H "Content-Type: application/json" \\
      -d '{"readings": [{"time": "2026-09-23T10:00:00Z", "value_kW": 123.4}]}'
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from pydantic import BaseModel, Field, field_validator

API_URL = os.environ.get("API_URL", "http://predikce-api:8000/predict")
HISTORY_HOURS = int(os.environ.get("HISTORY_HOURS", "48"))
MIN_HISTORY_STEPS = int(os.environ.get("MIN_HISTORY_STEPS", "192"))  # 48h @ 15min - stejny limit jako API
METER_NAME = os.environ.get("METER_NAME", "T12 P1.1")
RESAMPLE_MIN = 15

INFLUX_URL = os.environ["INFLUX_URL"]
INFLUX_TOKEN = os.environ["INFLUX_TOKEN"]
INFLUX_ORG = os.environ["INFLUX_ORG"]
INFLUX_BUCKET = os.environ["INFLUX_BUCKET"]

influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx.write_api(write_options=SYNCHRONOUS)
query_api = influx.query_api()


class Reading(BaseModel):
    time: datetime = Field(..., description="Casova znacka mereni v UTC")
    value_kW: float

    @field_validator("time")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("cas musi mit explicitni casove pasmo (UTC)")
        return v.astimezone(timezone.utc)


class IngestRequest(BaseModel):
    readings: list[Reading] = Field(..., min_length=1)


app = FastAPI(title="Predikce - ingest do InfluxDB")


@app.get("/health")
def health():
    return {"status": "ok"}


def write_skutecnost(readings: list[Reading]) -> int:
    points = [
        Point("skutecnost").tag("meter", METER_NAME).field("hodnota_kW", r.value_kW).time(r.time)
        for r in readings
    ]
    write_api.write(bucket=INFLUX_BUCKET, record=points)
    return len(points)


def query_recent_history() -> pd.DataFrame:
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{HISTORY_HOURS}h)
      |> filter(fn: (r) => r._measurement == "skutecnost")
      |> filter(fn: (r) => r.meter == "{METER_NAME}")
      |> filter(fn: (r) => r._field == "hodnota_kW")
      |> sort(columns: ["_time"])
    '''
    df = query_api.query_data_frame(flux, org=INFLUX_ORG)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame(columns=["_time", "_value"])
    if df.empty:
        return pd.DataFrame(columns=["time", "value_kW"])
    out = df[["_time", "_value"]].rename(columns={"_time": "time", "_value": "value_kW"})
    return out.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)


def write_predikce(predictions: list[dict], predicted_at: datetime) -> int:
    points = [
        Point("predikce")
        .tag("meter", METER_NAME)
        .tag("predicted_at", predicted_at.isoformat())
        .field("hodnota_kW", float(p["value_kW"]))
        .time(pd.Timestamp(p["time"]))
        for p in predictions
    ]
    write_api.write(bucket=INFLUX_BUCKET, record=points)
    return len(points)


@app.post("/ingest")
def ingest(req: IngestRequest):
    n_written = write_skutecnost(req.readings)

    history = query_recent_history()
    if len(history) < MIN_HISTORY_STEPS:
        return {
            "stored_readings": n_written,
            "history_available": len(history),
            "prediction_made": False,
            "reason": f"nedostatek historie v Influxu ({len(history)} < {MIN_HISTORY_STEPS} zaznamu = 48h)",
        }

    payload = {
        "readings": [
            {"time": pd.Timestamp(t).isoformat(), "value_kW": float(v)}
            for t, v in zip(history["time"], history["value_kW"])
        ]
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"predikcni API nedostupne/selhalo: {e}")

    body = resp.json()
    predicted_at = datetime.now(timezone.utc)
    n_pred = write_predikce(body["predictions"], predicted_at)

    return {
        "stored_readings": n_written,
        "history_available": len(history),
        "prediction_made": True,
        "predicted_at": predicted_at.isoformat(),
        "predictions_written": n_pred,
        "model_run_id": body["model_run_id"],
        "documented_test_mae_kW": body["documented_test_mae_kW"],
    }
