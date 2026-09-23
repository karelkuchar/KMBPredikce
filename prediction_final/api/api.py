"""
Python API pro 24h predikci: klient posle historicka mereni Avg.3P[kW],
API vrati 96 kroku (24h, 15min) predikce dopredu.

Zamerne izolovane - importuje jen `common` (VLASTNI kopie, ve stejne
slozce `prediction_final/api/`), nic mimo tuto slozku (krome natrenovanych
modelu v `../runs/`). Nacte nejnovejsi natrenovany model podle
`runs/latest.json` jednou pri startu (ne pri kazdem pozadavku).

Vstupni pozadavky (viz Reading/PredictRequest nize):
- Casy MUSI byt v UTC (explicitni casove pasmo v ISO 8601, napr. '...Z'
  nebo '+00:00') - naivni datetime bez pasma je odmitnuto.
- Presne na 15min mrizce (minuta deleni 15, 0 sekund/mikrosekund).
- Serazeno vzestupne, bez duplicit.
- Minimalne MIN_HISTORY_STEPS zaznamu (48h, viz UKOL.md "Klicove omezeni
  nasazeni") - min. co system potrebuje k rozumne predikci. Kratsi mezery
  (do MAX_GAP_FILL_STEPS) se linearne interpoluji, delsi jsou odmitnuty.

Pouziti (vyvoj):
    python3 -m uvicorn api:app --reload --app-dir prediction_final/api

Priklad pozadavku (POST /predict, zkraceno - ve skutecnosti min. 192 zaznamu):

    {
      "readings": [
        {"time": "2026-08-24T17:15:00+00:00", "value_kW": 202.54},
        {"time": "2026-08-24T17:30:00+00:00", "value_kW": 198.19},
        ... (celkem min. 192 zaznamu, 15min krok) ...
        {"time": "2026-08-31T00:00:00+00:00", "value_kW": 144.98}
      ]
    }

Priklad odpovedi (zkraceno):

    {
      "predictions": [
        {"time": "2026-08-31T00:15:00Z", "value_kW": 144.94},
        {"time": "2026-08-31T00:30:00Z", "value_kW": 144.558},
        ... (celkem 96 kroku) ...
        {"time": "2026-09-01T00:00:00Z", "value_kW": 158.038}
      ],
      "model_run_id": "20260922_215756",
      "documented_test_mae_kW": 19.367,
      "documented_test_mape_pct": 12.546,
      "input_readings": 604,
      "gaps_filled_steps": 0
    }

Plny priklad klienta: prediction_final/api/api_client_example.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from xgboost import XGBRegressor

import common

MIN_HISTORY_STEPS = 192  # 48h - viz UKOL.md "Klicove omezeni nasazeni"
MAX_GAP_STEPS = common.MAX_GAP_FILL_STEPS  # 24 kroku = 6h - stejny limit jako pri treninku
HORIZON_STEPS = 96  # 24h dopredu, 15min krok


# ============================================================
# Nacteni modelu (jednou, pri startu)
# ============================================================
def load_deployed_model():
    run_dir = common.latest_run_dir()
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(run_dir / "model.json"))
    return xgb_model, meta, run_dir.name


XGB_MODEL, METADATA, RUN_ID = load_deployed_model()


# ============================================================
# Pydantic modely (vstup/vystup + validace)
# ============================================================
class Reading(BaseModel):
    time: datetime = Field(..., description="Casova znacka mereni v UTC (napr. '2026-09-22T10:00:00Z')")
    value_kW: float = Field(..., description="Avg.3P[kW] - celkovy cinny vykon (soucet 3 fazi)")

    @field_validator("time")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("cas musi mit explicitni casove pasmo (UTC) - napr. '...Z' nebo '+00:00', ne naivni datetime")
        return v.astimezone(timezone.utc)

    @field_validator("time")
    @classmethod
    def _require_15min_grid(cls, v: datetime) -> datetime:
        if v.minute % common.RESAMPLE_MIN != 0 or v.second != 0 or v.microsecond != 0:
            raise ValueError(f"cas musi byt presne na {common.RESAMPLE_MIN}minutove mrizce (napr. :00, :15, :30, :45), dostal jsem {v.isoformat()}")
        return v

    @field_validator("value_kW")
    @classmethod
    def _require_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("value_kW musi byt konecne cislo (ne NaN/Inf)")
        return v


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(
        ..., min_length=1,
        description=(
            f"Historicka mereni Avg.3P[kW], serazena vzestupne podle casu (nebo API sama seradi), "
            f"krok presne {common.RESAMPLE_MIN} min, minimalne {MIN_HISTORY_STEPS} zaznamu (48h). "
            f"Delsi historie (az 7 dni = 672 zaznamu) zlepsuje presnost tydenniho lagu."
        ),
    )
    include_history: bool = Field(
        False,
        description=(
            "Pokud true, odpoved bude navic obsahovat pole 'history' - historii tak, "
            "jak ji API skutecne pouzilo (serazenou, s doplnenymi mezerami). Vychozi false."
        ),
    )


class PredictionPoint(BaseModel):
    time: datetime
    value_kW: float


class PredictResponse(BaseModel):
    predictions: list[PredictionPoint]
    model_run_id: str
    documented_test_mae_kW: float
    documented_test_mape_pct: float
    input_readings: int
    gaps_filled_steps: int
    history: list[PredictionPoint] | None = Field(
        None, description="Pouze pokud byl v pozadavku 'include_history: true' - pouzita historie (serazena, doplnena)."
    )


class ErrorDetail(BaseModel):
    detail: str


# ============================================================
# Priprava vstupu (validace, doplneni mezer) + vystavba FeatureSet
# ============================================================
def prepare_history(readings: list[Reading]) -> tuple[pd.DataFrame, int]:
    """Vraci (serazeny/doplneny df se sloupci time+TARGET_COL, pocet doplnenych kroku)."""
    times = [r.time for r in readings]
    values = [r.value_kW for r in readings]
    df = pd.DataFrame({"time": times, common.TARGET_COL: values}).sort_values("time")

    if df["time"].duplicated().any():
        dupes = [t.isoformat() for t in df.loc[df["time"].duplicated(), "time"].head(5)]
        raise ValueError(f"duplicitni casove znacky ve vstupu: {dupes}{' ...' if len(dupes) == 5 else ''}")

    if len(df) < MIN_HISTORY_STEPS:
        raise ValueError(
            f"nedostatek historie: dostal jsem {len(df)} zaznamu, potrebuji minimalne "
            f"{MIN_HISTORY_STEPS} ({MIN_HISTORY_STEPS * common.RESAMPLE_MIN // 60}h) - "
            f"viz UKOL.md 'Klicove omezeni nasazeni'."
        )

    full_index = pd.date_range(df["time"].iloc[0], df["time"].iloc[-1], freq=f"{common.RESAMPLE_MIN}min")
    missing = full_index.difference(df["time"])
    if len(missing) > 0:
        # najdi nejdelsi souvislou mezeru
        missing_sorted = missing.sort_values()
        max_gap = 1
        cur_gap = 1
        for i in range(1, len(missing_sorted)):
            if missing_sorted[i] - missing_sorted[i - 1] == pd.Timedelta(minutes=common.RESAMPLE_MIN):
                cur_gap += 1
                max_gap = max(max_gap, cur_gap)
            else:
                cur_gap = 1
        if max_gap > MAX_GAP_STEPS:
            raise ValueError(
                f"mezera ve vstupnich datech je prilis dlouha ({max_gap} kroku = "
                f"{max_gap * common.RESAMPLE_MIN / 60:.1f}h), maximalni akceptovana mezera je "
                f"{MAX_GAP_STEPS} kroku ({MAX_GAP_STEPS * common.RESAMPLE_MIN / 60:.0f}h)."
            )

    df = df.set_index("time").reindex(full_index)
    df.index.name = "time"
    df[common.TARGET_COL] = df[common.TARGET_COL].interpolate(limit=MAX_GAP_STEPS)
    df = df.reset_index()
    return df, int(len(missing))


def build_live_feature_set(history_df: pd.DataFrame) -> tuple[common.FeatureSet, int]:
    """history_df: sloupce 'time' + common.TARGET_COL, souvisla 15min mrizka, bez mezer.
    Rozsiri o HORIZON_STEPS budoucich radku (jen kalendarni featury, cilova hodnota neni
    potreba - autoregressive_predict ji nikdy necte pro radky > t0), vrati (FeatureSet, t0)."""
    df = history_df.copy()
    last_time = df["time"].iloc[-1]
    future_times = pd.date_range(
        last_time + pd.Timedelta(minutes=common.RESAMPLE_MIN),
        periods=HORIZON_STEPS, freq=f"{common.RESAMPLE_MIN}min",
    )
    future_df = pd.DataFrame({"time": future_times, common.TARGET_COL: np.nan})
    full_df = pd.concat([df, future_df], ignore_index=True)

    time_fut = full_df["time"] + pd.Timedelta(minutes=common.RESAMPLE_MIN)
    tf = common.time_feats(time_fut)
    for c in tf.columns:
        full_df[c] = tf[c]
    full_df["is_break"] = common.is_break_or_holiday(time_fut)

    t0 = len(df) - 1  # posledni znamy (realny) radek - odtud rekurze zacina
    fs = common.FeatureSet(df=full_df, feature_cols=METADATA["feature_cols"], lag_map=METADATA["lag_map"])
    return fs, t0


# ============================================================
# FastAPI aplikace
# ============================================================
app = FastAPI(
    title="Predikce spotřeby T12 P1.1",
    description=(
        "Prijme historicka mereni Avg.3P[kW] a vrati 24h (96 kroku, 15min) rekurzivni "
        "predikci. Model: univariate + is_break (VUT semestr/svatky) + blend XGBoost/"
        "SeasonalNaive24h - viz prediction/EXPERIMENT_LOG.md 'Kolo 3'."
    ),
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """FastAPI/Starlette default handler crashes (500) misto vraceni 422,
    pokud chybne vstupni JSON obsahovalo NaN/Inf - Pydantic ho spravne
    zamitne, ale pak se ho pokusi ozvenit zpet v chybove odpovedi a
    Starlette JSONResponse pouziva allow_nan=False (striktni JSON nezna
    NaN literal). Tady se neserializovatelne hodnoty pred odpovedi ocisti."""
    def sanitize(v):
        if isinstance(v, float) and not np.isfinite(v):
            return str(v)
        if isinstance(v, BaseException):
            return str(v)
        return v

    errors = []
    for err in exc.errors():
        err = {k: v for k, v in err.items() if k != "ctx"}  # ctx.error muze nest surovy ValueError - msg uz stejnou info ma citelne
        err["input"] = sanitize(err.get("input"))
        errors.append(err)
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_run_id": RUN_ID,
        "documented_test_mae_kW": METADATA["test_mae_24h_kW"],
        "documented_test_mape_pct": METADATA["test_mape_24h_pct"],
        "blend_alpha": METADATA["blend_alpha"],
        "min_history_steps": MIN_HISTORY_STEPS,
        "horizon_steps": HORIZON_STEPS,
    }


@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorDetail}})
def predict(req: PredictRequest):
    try:
        history_df, gaps_filled = prepare_history(req.readings)
        fs, t0 = build_live_feature_set(history_df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    preds = common.blend_autoregressive_predict(
        XGB_MODEL, fs, n_test=HORIZON_STEPS, alpha=METADATA["blend_alpha"],
        reset_steps=None, t0=t0,
    )
    pred_times = fs.df["time"].iloc[t0 + 1: t0 + 1 + HORIZON_STEPS].tolist()

    history_out = None
    if req.include_history:
        history_out = [
            PredictionPoint(time=t, value_kW=round(float(v), 3))
            for t, v in zip(history_df["time"], history_df[common.TARGET_COL])
        ]

    return PredictResponse(
        predictions=[PredictionPoint(time=t, value_kW=round(float(p), 3)) for t, p in zip(pred_times, preds)],
        model_run_id=RUN_ID,
        documented_test_mae_kW=METADATA["test_mae_24h_kW"],
        documented_test_mape_pct=METADATA["test_mape_24h_pct"],
        input_readings=len(req.readings),
        gaps_filled_steps=gaps_filled,
        history=history_out,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
