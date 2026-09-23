# prediction_final

Finální, samostatná verze predikčního modelu spotřeby (elektroměr T12 P1.1).
Funguje nezávisle na zbytku repozitáře — dá se zkopírovat a použít
samostatně. Detailní zdůvodnění a všechna srovnání jsou v
`prediction/EXPERIMENT_LOG.md`, tady jen shrnutí principu.

## Co se predikuje

- **Cílová veličina:** `Avg.3P[kW]` — celkový (součtový) okamžitý činný
  výkon přes všechny 3 fáze, v kW.
- **Granularita:** krok 15 minut.
- **Horizont:** 24 hodin dopředu = 96 kroků.

## Přístup

- **Jeden rekurzivní (autoregresivní) model** pro celý 24h horizont. Model
  se naučí predikovat vždy jen **1 krok dopředu** (`y[t+15min]` z featur v
  čase `t`).
- Pro 24h výhled se tenhle 1-krokový model aplikuje **96× po sobě**:
  predikce kroku 1 se použije jako vstup (lag/rolling featura) pro predikci
  kroku 2, atd. Chyba se tím může kumulovat, proto se přesnost vyhodnocuje i
  po jednotlivých krocích horizontu, ne jen v celkovém průměru.
- Model umí predikovat už po **48 h** běžícího sběru dat od startu/restartu —
  featury proto používají hlavně lagy ≤ 24 h, s jednou výjimkou (viz níž).

## Vstupní featury (feature-set "standardní" / `univariate` + `is_break`)

- **Lagy cíle** (`Avg.3P[kW]` v minulosti): 15, 30, 60, 120, 180, 360, 720,
  1440 min (=24h) zpátky.
- **Klouzavé statistiky** cíle: mean/std v oknech 1h, 4h, 8h.
- **Delta** (rozdíl oproti předchozímu kroku).
- **`lag_1week`** (7 dní zpátky) — jediná výjimka z pravidla "≤24h": při
  běhu s méně než 7 dny historie se použije nejstarší dostupná hodnota jako
  náhrada za chybějící lag.
- **Kalendářní featury cílového času** (t+15min): hodina, minuta, den v
  týdnu, `sin`/`cos` kódování denního slotu / měsíce / dne v roce, víkend.
- **`is_break`** — budova je pracoviště VUT, spotřeba se řídí letním
  provozem/prázdninami a státními svátky, ne jen dnem v týdnu (přidáno v
  kole 3 na základě upřesnění od uživatele). Viz tabulka níž.
- **Žádné exogenní veličiny.** Původně (kolo 1/2) se testovaly i exogenní
  featury specifické pro tento síťový analyzátor (`Avg.Q3[kvar]`,
  `Avg.THDI1-3[%]`, `Min/Max.P1[kW]`, `Min/Max.P3[kW]`), s hodnotou zmrazenou
  na poslední známý údaj během 24h rekurze. Po opravě bufferového bugu v
  rekurzivní inferenci (viz `prediction/EXPERIMENT_LOG.md` sekce "Oprava
  kritického bufferového bugu") se ukázalo, že samotná (opravená) historie
  cíle nese dost signálu sama o sobě a exogenní featury přidávají jen šum —
  proto tahle finální verze žádné nepoužívá.

## Úplný seznam vstupních featur (28 celkem)

### Kalendářní (11) — z cílového času `t+15min`, ne z aktuálního `t`

| feature | jak vzniká |
|---|---|
| `hour` | hodina cílového času |
| `minute` | minuta cílového času |
| `dayofweek` | den v týdnu cílového času (0=pondělí … 6=neděle) |
| `day_sin` | `sin(2π · slot/96)`, kde `slot = hour*4 + minute//15` (pozice v rámci dne, 0–95) |
| `day_cos` | `cos(2π · slot/96)` |
| `month_sin` | `sin(2π · month/12)` |
| `month_cos` | `cos(2π · month/12)` |
| `year_sin` | `sin(2π · den_v_roce/365.25)` |
| `year_cos` | `cos(2π · den_v_roce/365.25)` |
| `is_weekend` | 1 pokud `dayofweek ≥ 5` (sobota/neděle), jinak 0 |
| `is_break` | 1 pokud je cílový den v červenci/srpnu (hlavní prázdniny) NEBO je to český státní svátek (fixní data + Velikonoční pondělí dopočítané přes `dateutil.easter`), jinak 0 |

### Lagy cíle (8) — hodnota `Avg.3P[kW]` v minulosti, vzhledem k aktuálnímu `t`

| feature | jak vzniká |
|---|---|
| `lag_Sum_15m` | hodnota před 15 min (1 krok zpět) |
| `lag_Sum_30m` | hodnota před 30 min (2 kroky) |
| `lag_Sum_60m` | hodnota před 60 min (4 kroky) |
| `lag_Sum_120m` | hodnota před 120 min (8 kroků) |
| `lag_Sum_180m` | hodnota před 180 min (12 kroků) |
| `lag_Sum_360m` | hodnota před 360 min (24 kroků) |
| `lag_Sum_720m` | hodnota před 720 min (48 kroků) |
| `lag_Sum_1440m` | hodnota před 1440 min = 24h (96 kroků) |

### Klouzavé statistiky (6) — počítané z posledních N kroků `Avg.3P[kW]` včetně `t`

| feature | jak vzniká |
|---|---|
| `roll_mean_4s` | průměr za posledních 4 kroky (1h) |
| `roll_std_4s` | směrodatná odchylka za stejné okno (1h) |
| `roll_mean_16s` | průměr za posledních 16 kroků (4h) |
| `roll_std_16s` | směrodatná odchylka za stejné okno (4h) |
| `roll_mean_32s` | průměr za posledních 32 kroků (8h) |
| `roll_std_32s` | směrodatná odchylka za stejné okno (8h) |

### Delta + týdenní/denní lag (3)

| feature | jak vzniká |
|---|---|
| `delta_1s` | `Avg.3P[kW]` v čase `t` mínus hodnota v čase `t-15min` — trend za poslední krok |
| `lag_1week` | hodnota přesně 7 dní (672 kroků) zpátky, stejný čas dne — týdenní sezónnost |
| `lag_1day` | hodnota přesně 24h (96 kroků) zpátky — denní sezónnost |

## Model

Finální predikce je **blend dvou modelů**, ne jen samotný XGBoost:

```
predikce = alpha × XGBoost + (1 − alpha) × SeasonalNaive24h
```

- **XGBoost** (regrese, `reg:absoluteerror`, výchozí hyperparametry —
  hyperparameter tuning se v kole 3 zkoušel, ale v kombinaci s hledáním
  `alpha` na stejném 14denním validačním okně vedl k výraznému přeučení
  (skvělé číslo na validaci, ale mnohem horší na testu) — viz
  `prediction/EXPERIMENT_LOG.md` "Kolo 3" pro detaily. Výchozí hyperparametry
  jsou proto bezpečnější volba.
- **SeasonalNaive24h** — triviální baseline, predikce = hodnota přesně před
  24h (`lag_1day`). Sama o sobě výrazně horší než XGBoost, ale její chyby
  jsou jiného druhu (čistá sezónnost, žádné učení) — vážený průměr obou
  modelů snižuje rozptyl/drift dlouhé 24h rekurze (shrinkage efekt).
- **`alpha`** (aktuálně 0.8) se hledá na validaci (14 dní), zkouší se
  0.0–1.0 po kroku 0.1, viz `train.py`.
- **Důležitá implementační poznámka:** XGBoost a SeasonalNaive24h běží jako
  **dvě zcela nezávislé 24h rekurze** (každá se svým vlastním bufferem) —
  teprve jejich výstupní pole se na konci zprůměrují
  (`common.blend_autoregressive_predict`). Zprůměrovat je *uvnitř* jedné
  sdílené rekurze (tj. použít smíchanou hodnotu jako zpětnovazební vstup do
  bufferu) dává výrazně horší výsledek, protože by pak XGBoost viděl ve
  svých lag featurách směs vlastní a sezónní predikce místo vlastní čisté
  trajektorie, na kterou byl natrénovaný — tahle chyba se objevila při prvním
  nasazení blendu a je zdokumentovaná v `EXPERIMENT_LOG.md` jako ponaučení.
- **Aktuální natrénovaný model:** 24h rekurzivní test MAE = 19.37 kW,
  RMSE = 26.97 kW, MAPE = 12.55 % — o ~32 % lepší než baseline hned po
  opravě bufferového bugu (28.5-28.9 kW), postupně přes blend (21.14 kW) a
  přidání `is_break` (19.37 kW). Viz `runs/latest.json` a
  `runs/<run_id>/metadata.json` pro přesná čísla aktuálního běhu.

## Data

- Zdroj: `data_processed/main_archive_15min.csv` (1min data z analyzátoru,
  resamplovaná na 15min).
- Používá se data **od 2025-01-01** — dřívější úsek byl označen jako
  měření s chybou / hodně chybějícími hodnotami.

## Soubory ve složce

| soubor / složka | účel |
|---|---|
| `common.py` | sdílené jádro — načtení dat, feature engineering (vč. `is_break_or_holiday`/český kalendář svátků), rekurzivní inference, `SeasonalNaiveModel`/`blend_autoregressive_predict` (importují ho `train.py`, `predict.py`, `test/run_test.py` i `api.py`) |
| `train.py` | trénink XGBoostu (výchozí hyperparametry) + hledání `alpha` pro blend se SeasonalNaive24h + uložení do vlastní datované složky |
| `predict.py` | testovací skript — načte nejnovější uložený model a vypíše 24h predikci vs. skutečnost pro první cyklus testovacího okna |
| `test/run_test.py` | testovací skript — spustí nejnovější model na **celém** testovacím okně a uloží `test_results.csv` (skutečnost + predikce, celé okno), `test_plot.png` (graf) a `test_metrics_breakdown.csv` (MAE/RMSE/MAPE rozepsané po hodině dne, dni a měsíci). Ukládá **na obě místa** — do `test/` (vždy "poslední výsledek") i do příslušné `runs/<run_id>/` (svázané s konkrétním během, historie se nepřepisuje) |
| `api.py` | **produkční REST API** (FastAPI) — `POST /predict` přijme historická měření, vrátí 24h predikci; `GET /health`. Viz sekce "API" níž. |
| `api_client_example.py` | minimální příklad klienta — přesný tvar požadavku/odpovědi |
| `test/test_api.py` | end-to-end test API (spustí skutečný server, ověří shodu s přímým výpočtem, "delší běh" přes 25 bodů v testovacím okně, validaci vstupu) |
| `runs/latest.json` | ukazatel na nejnovější trénovací běh |
| `runs/<YYYYMMDD_HHMMSS>/` | výstup jednoho běhu `train.py` — viz níž |

### Obsah jedné složky `runs/<YYYYMMDD_HHMMSS>/`

Každé spuštění `train.py` vytvoří **novou** složku pojmenovanou datem a
časem běhu — nic se nepřepisuje, historie běhů zůstává zachovaná. Následné
spuštění `test/run_test.py` do stejné složky doplní i testovací výsledky.

| soubor | účel |
|---|---|
| `model.json` | natrénovaný XGBoost model (jen XGBoost složka blendu — SeasonalNaive24h se nic netrénuje) |
| `best_hyperparams.json` | hyperparametry XGBoostu + `blend_alpha` (vítězná váha blendu z validace), samostatně a čitelně |
| `metadata.json` | pořadí/názvy feature sloupců, lag mapa, `blend_alpha`, dosažené metriky, čas běhu — potřeba pro správné načtení modelu v `predict.py`/`test/run_test.py`/`api.py` |
| `train.log` | kompletní konzolový výstup běhu (včetně alpha-search tabulky) |
| `test_results.csv` | *(doplní `test/run_test.py`)* skutečnost + predikce po jednotlivých krocích, celé test okno |
| `test_plot.png` | *(doplní `test/run_test.py`)* graf skutečnost vs. predikce |
| `test_metrics_breakdown.csv` | *(doplní `test/run_test.py`)* MAE/RMSE/MAPE rozepsané po hodině dne (0–23), kalendářním dni a měsíci |

## API

`api/api.py` — FastAPI server, `POST /predict` přijme historická měření
`Avg.3P[kW]` a vrátí 96 kroků (24h) predikce dopředu. Celá složka `api/` je
samostatná — má **vlastní kopii** `common.py` (ne sdílenou s `prediction_final/`
výše), takže se dá zkopírovat/nasadit nezávisle na zbytku repozitáře
(kromě natrénovaných modelů v `../runs/`).

**Vstup** (`POST /predict`, tělo `{"readings": [...]}`):

| pravidlo | proč |
|---|---|
| Časy **musí mít explicitní časové pásmo** (ISO 8601, např. `...Z` nebo `+00:00`) — naivní datetime bez pásma je odmítnut | jednoznačnost, žádné hádání lokálního pásma. Vše se okamžitě převede na **UTC**, takže přechod mezi letním/zimním časem (CEST/CET) nehrozí — UTC žádný DST posun nemá |
| Přesně na 15minutové mřížce (`:00`, `:15`, `:30`, `:45`) | odpovídá granularitě modelu |
| Seřazeno vzestupně, bez duplicit | API si samo seřadí, ale duplicitní časy odmítne |
| Minimálně **192 záznamů (48 h)** | `UKOL.md` "Klíčové omezení nasazení" — méně systém nemá umět |
| Mezery do 6 h (24 kroků) se lineárně doplní (`gaps_filled_steps` v odpovědi), delší jsou odmítnuty (400) | kontrola úplnosti dat — stejný limit jako při přípravě trénovacích dat |
| Hodnoty musí být konečná čísla (ne NaN/Inf) | |
| `include_history: true/false` (volitelné, výchozí `false`) | pokud `true`, odpověď navíc obsahuje pole `history` — historii přesně tak, jak ji API použilo (seřazenou, s doplněnými mezerami) |

Delší historie (až 7 dní = 672 kroků) zlepšuje přesnost týdenního lagu.
Ověřeno i na datech přesahujících skutečný přechod letního/zimního času
(`api/test_api.py::test_dst_transition`) a na datech s mezerou nad/pod
povoleným limitem (`test_gap_handling`).

**Výstup:** 96 dvojic `{time, value_kW}` (UTC, 15min krok) + `model_run_id`,
dokumentovaná testovací MAE/MAPE, počet doplněných kroků mezer, volitelně
použitá historie (viz `include_history` výše).

**Chybové odpovědi:** `400` (validace na úrovni dat — krátká historie,
duplicity, moc velká mezera) nebo `422` (validace na úrovni typů/formátu —
Pydantic). Nikdy `500` na vadný vstup — ověřeno `api/test_api.py`
(dva bugy nalezené a opravené při testování zdokumentované v
`prediction/EXPERIMENT_LOG.md` sekce "Python API pro predikci").

**Příklad požadavku** (zkráceno — ve skutečnosti min. 192 záznamů):

```json
{
  "readings": [
    {"time": "2026-08-24T17:15:00+00:00", "value_kW": 202.54},
    {"time": "2026-08-24T17:30:00+00:00", "value_kW": 198.19},
    "... (celkem min. 192 záznamů, 15min krok) ...",
    {"time": "2026-08-31T00:00:00+00:00", "value_kW": 144.98}
  ]
}
```

**Příklad odpovědi** (zkráceno — ve skutečnosti 96 kroků):

```json
{
  "predictions": [
    {"time": "2026-08-31T00:15:00Z", "value_kW": 144.94},
    {"time": "2026-08-31T00:30:00Z", "value_kW": 144.558},
    "... (celkem 96 kroků) ...",
    {"time": "2026-09-01T00:00:00Z", "value_kW": 158.038}
  ],
  "model_run_id": "20260922_215756",
  "documented_test_mae_kW": 19.367,
  "documented_test_mape_pct": 12.546,
  "input_readings": 604,
  "gaps_filled_steps": 0
}
```

```bash
python3 -m uvicorn api:app --app-dir prediction_final/api --port 8000   # spustit server
python3 prediction_final/api/api_client_example.py                       # ukázkové volání (server musí běžet)
```

Interaktivní dokumentace (Swagger UI) po spuštění na `http://127.0.0.1:8000/docs`.

Plné (nezkrácené) skutečné příklady požadavku/odpovědi — krátký (minimální
192 záznamů), delší (604 záznamů) a s `include_history: true` — jsou v
`api/example_json/`.

## Spuštění

```bash
python3 prediction_final/train.py            # natrénuje XGBoost, najde alpha pro blend, uloží do runs/<datum_cas>/
python3 prediction_final/predict.py          # načte nejnovější model, vypíše testovací 24h predikci
python3 prediction_final/test/run_test.py    # spustí nejnovější model na celém test okně, uloží CSV + graf
python3 prediction_final/api/test_api.py     # end-to-end test API (spustí vlastní server, ověří vše výše)
```

**Délka testovacího okna** se nastavuje proměnnou `TEST_DAYS` nahoře v
`train.py` (výchozí 30 dní) — pro jiný běh (např. 60denní ověření, viz
kolo 2/3) stačí ji tam přepsat a spustit `train.py` znovu. `predict.py` a
`test/run_test.py` si při načtení modelu automaticky přeberou stejnou
délku okna z `metadata.json` daného běhu, takže zůstanou konzistentní.
