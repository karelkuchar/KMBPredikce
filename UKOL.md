# Projekt: predikce spotřeby (kmb_predikce)

## Zadání (finální)
- **Cíl:** predikovat `Avg.3P[kW]` (= součet výkonu tří fází, "Sum_P_kW") z dat elektroměru T12 P1.1.
- **Krok:** 15 minut. **Horizont:** 24 hodin dopředu (96 kroků).
- **Přístup: jeden autoregresivní model** (ne 96 samostatných modelů na horizont). Model predikuje vždy jeden krok dopředu `y[t+15min]` z featur v čase `t`. Pro 24h výhled (96 kroků) se aplikuje **rekurzivně**: predikce kroku 1 se použije jako (částečně) "pozorovaná" hodnota pro výpočet lag/rolling featur při predikci kroku 2, atd. — klasický recursive multi-step forecasting. Potvrzeno uživatelem (2. upřesnění).
  - Důsledek: chyba se může kumulovat přes 96 kroků → v `srovnani_1` nutně vyhodnotit chybu **per horizont** (1..96), abychom viděli, jak rychle přesnost degraduje.
- **Klíčové omezení nasazení:** systém musí umět predikovat už po **48 h** běžícího live sběru dat od nasazení/restartu.
- **Model:** po opravě bufferového bugu a kole 3 (viz checklist níž) vychází nejlépe **`univariate` + `is_break` (VUT semestr/svátky, bez exogenních featur) + blend 0.8×XGBoost + 0.2×SeasonalNaive24h**, 24h MAE = 19.37 kW (viz `prediction/EXPERIMENT_LOG.md` sekce "Kolo 3"). CatBoost i LightGBM otestovány, XGBoost vede jako základní model blendu.

## Stav dat
- `data/T12 P1.1/` — surová data z analyzátoru sítě (ProBEMS/Janitza export), měsíční CSV 2024-09 až 2026-08:
  - **Main Archive** — 1min interval, napětí/proud/výkon/THD atd. → **jediný zdroj** (target + features), Electricity Meter/Histogram/Log se zatím nepoužívají.
- `data_processed/main_archive_15min.csv` — sjednocený dataset, 1min data spojena přes 23 měsíčních souborů (2024-09-21 .. 2026-09-01), resamplována na 15min (Avg.* = mean, Min.*/Max.* = min/max v okně). Skript: `data_processed/build_dataset.py`.
  - Drobné mezery: 2x DST přechod (~45min), pár hodinových výpadků (leden/květen 2026), a větší mezera na začátku (2024-09-29 až 2024-12-02 — zřejmě instalace/testování přístroje, ne ostrý provoz).
- **Kolo 2 update:** modelovaný úsek posunut z `2024-12-02` na **`2025-01-01`** (`pipeline.py::MODEL_START`) — uživatel označil období 2024-12 za chybu měření / hodně chybějících dat (domenová znalost, mimo to co je vidět jako NaN-mezery v `main_archive_15min.csv`). Všechny kolo 2 výsledky (`srovnani_2/`) jsou na tomto novém, kratším okně; kolo 1 (`srovnani_1/`) zůstává na starém okně (2024-12-02+) — **čísla mezi koly nejsou 1:1 srovnatelná**, srovnani_2/run_comparison.py ale běží na stejném harness i pro univariate/multivariate, takže funguje i jako aktualizovaný baseline.

## Feature set (dle zadání uživatele, zobecněno na 96 horizontů)

Target: `Avg.3P[kW]`.

**Skupina A — lagy targetu, čas t** (vše ≤24h, OK pro 48h cold-start):
`lag_15m`(1 krok) `lag_30m`(2) `lag_60m`(4) `lag_120m`(8) `lag_180m`(12) `lag_360m`(24) `lag_720m`(48) `lag_1440m`(96=24h)

**Skupina B — odvozené statistiky, čas t:**
`roll_mean_4`/`roll_std_4` (1h okno), `roll_mean_16`/`roll_std_16` (4h okno), `roll_mean_32`/`roll_std_32` (8h okno), `delta_1` (Δ oproti předchozímu kroku), `lag_1day` (96 kroků, duplicitní s lag_1440m — ponecháno pro parity s originálním zadáním), a `lag_1week` (672 kroků = 7 dní).
- `lag_1week` koliduje s pravidlem "funkční už po 48h" → v rekurzivní evaluaci se při resyncu (7d/24h reset) přepočítá z reálných dat, jinak zůstává poslední známá/predikovaná hodnota v bufferu (viz implementace níže — ne NaN, buffer-based jako u ostatních lagů).
- (pozn.: `week_sin`/`week_cos` z dřívější verze zadání jsem vypustil — ve skutečném referenčním skriptu uživatele nejsou, dayofweek/day_sin/day_cos ze skupiny C periodicitu týdne/dne pokrývají dostatečně.)

**Skupina C — kalendářní featury cílového času t+15min (jeden krok dopředu):**
`hour` `minute` `dayofweek` `day_sin`/`day_cos` (slot dne 0-95) `month_sin`/`month_cos` `year_sin`/`year_cos` (den v roce) `is_weekend`

**Exogenní featury (jen multivariate varianta):** `Avg.Q3[kvar]`, `Avg.THDI1-3[%]`, `Min/Max.P1[kW]`, `Min/Max.P3[kW]` — hodnota v čase t. Při rekurzi (mimo resync bod) se **zmrazí** na poslední reálně známé hodnotě, protože budoucí hodnoty těchto veličin nejsou při live nasazení dopředu známé (na rozdíl od targetu, který se autoregresivně dopočítává z vlastních predikcí).
- **Update po opravě bufferového bugu:** exogenní featury se ukázaly nepotřebné/škodlivé — `prediction_final/` teď používá jen skupiny A/B/C (žádné exogenní), viz checklist níž.

## Referenční implementace
Uživatel poskytl svůj původní fungující skript (LightGBM, jen `Sum_P_kW`, `autoregressive_predict` s reset-módy 24h/7d/bez-resetu). Přebíráno 1:1 (stejná definice lagů/rolling/delta/time_feats i logika bufferu), rozšířeno o multi-model + multivariate variantu + rozpad chyby po horizontech. Kód: `prediction/pipeline.py` (sdílené jádro), `prediction/srovnani_1/run_comparison.py` (orchestrace, běží zatím).

## Modelovací přístup
- **Trénink:** jeden model, jeden krok dopředu (`y[t+15min]` z featur v `t`), trénovaný na historii od 2024-12-02 (po instalačním období).
- **Inference na 24h dopředu:** rekurzivní smyčka — 96× po sobě: spočítat featury z aktuálního bufferu (real data + dosavadní vlastní predikce), predikovat další krok, přidat predikci do bufferu, opakovat.
- **Split:** časový (ne náhodný) — train / val (14 dní) / test (kolo 1: 30 dní, kvůli rychlosti rekurzivního běhu — lze později zvětšit).
- **Vyhodnocovací módy:** `1step` (rychlý vektorizovaný baseline bez rekurze — horní hranice přesnosti), `24h` (reset každých 96 kroků — **odpovídá produkčnímu nasazení**), `7d` (reset každých 672 kroků — stress test).
- Výstupy testů: `prediction/srovnani_1/*.csv` (metriky MAE/RMSE/MAPE per feature-set × model × mód, plus rozpad **per krok horizontu 1..96** pro mód `24h`).
- Finální výstup: shrnutí/doporučení (feature set + model + hyperparametry) v `prediction/`.

## Budoucí rozšíření (mimo aktuální kolo, ale ovlivňuje architekturu už teď)
- Export natrénovaného modelu + Python API, které přijme jen nutná vstupní data (poslední buffer měření), vrátí hodnotu predikce (24h dopředu, 15min krok) a volitelně (konfigurovatelně) lokálně uloží vstup i výstup.
- Samostatný klientský skript na straně měřiče, který bude API pravidelně volat.
- **Důsledek pro dnešní kód:** feature engineering (`build_features`) a rekurzivní inference (`autoregressive_predict`) v `prediction/pipeline.py` jsou už teď psané jako importovatelné funkce nezávislé na CLI/skriptu — API je pak jen tenká vrstva okolo stejného kódu, žádný přepis.

## Vyřešené otázky
- Target = součet 3 fází (Avg.3P[kW]) — potvrzeno.
- Výstup sjednocených dat = CSV (ne parquet) — potvrzeno.
- Electricity Meter (kWh) data — zatím NEzahrnuto, jede se jen z Main Archive (kW).
- Jeden autoregresivní model (rekurzivní), ne 96 modelů na horizont — potvrzeno.

## Otevřené otázky (doplň/oprav přímo zde)
- [ ] Existuje víc měřáků/budov kromě T12 P1.1, které by měly jít do stejného pipeline?

## Checklist / postup projektu

**Hotovo:**
- [x] Sjednocení dat → `data_processed/main_archive_15min.csv` (`data_processed/build_dataset.py`)
- [x] Feature engineering jádro (lagy/rolling/delta/time_feats, univariate="standardní" + multivariate="rozšířené") → `prediction/pipeline.py`
- [x] Rekurzivní (autoregresivní) inference harness s reset-módy → `prediction/pipeline.py`
- [x] **Kolo 1 hotovo** (`prediction/srovnani_1/`): LightGBM/CatBoost/XGBoost/Persistence/SeasonalNaive24h × standardní/rozšířené, módy 1step/24h/7d. Výsledky + interpretace: `prediction/EXPERIMENT_LOG.md` (sekce "Kolo 1"). Nejlepší: **rozšířené + XGBoost, MAE=28.47 kW (mód 24h)**. Klíčové zjištění: rozšířené featury (Q3/THDI/min-max P1,P3 — specifické pro tento analyzátor) výrazně zlepšují přesnost oproti standardnímu postupu (jen Avg.3P) u všech modelů; čistě standardní ML modely jsou v 24h rekurzi dokonce horší než triviální baseline. (Pozn.: přepočítáno v kole 2 na novém datovém okně, viz níž — nové číslo 28.59 kW.)

**Kolo 2 — HOTOVO** (`prediction/srovnani_2/`, detaily a čísla v `prediction/EXPERIMENT_LOG.md` sekce "Kolo 2"):
1. [x] Rozšířené exogenní kanály (U1-3, I1-3, PF, THDU) otestovány (`multivariate_plus`) — **zavrženo**, v 24h/7d rekurzi škodí (víc zmrazených featur = větší drift), zůstáváme u `multivariate` (Q3, THDI1-3, Min/Max P1/P3).
2. [x] Fázové zkreslení opraveno — `pipeline.py::staggered_recursive_predict` (rozprostřené starty). Periodic per-horizont křivka byla artefakt hodiny dne, staggered je hladká/monotónní-ish.
3. [x] Hyperparameter tuning (`tune.py`, 25 random-search pokusů/model, tuning na 24h rekurzivní val MAE) — XGBoost vylepšen o -6.4 % (28.59 → 26.76 kW), CatBoost tuning nepomohl.
4. [x] Ověřeno na 60denním test okně (`test60.py`) — pořadí modelů stejné, výsledek robustní.
5. [x] Zapsáno do `EXPERIMENT_LOG.md`.
6. **Mimo plán, přidáno za běhu:** uživatel označil data před 2025-01-01 za chybu měření → `pipeline.py::MODEL_START` posunuto na `2025-01-01`, kolo 1 (`srovnani_1/`) přepočítáno na stejném okně kvůli srovnatelnosti.

**Oprava kritického bufferového bugu — HOTOVO** (detaily v `prediction/EXPERIMENT_LOG.md`
sekce "Oprava kritického bufferového bugu"):
- Nalezeno při `/code-review` nad `prediction_final/`: AR buffer v `autoregressive_predict`
  chyběl o 1 hodnotu, kazil lag/rolling featury po 95 z 96 kroků každého cyklu (ne jen první
  krok). Bug byl přítomný od kola 1, ve 4 kopiích kódu (`pipeline.py`, `best_model.py`,
  `prediction_final/common.py`, `prediction_final/test/run_test.py`) — všechny opraveny a
  ověřeny instrumentací proti tréninkovým hodnotám.
- **Hlavní závěr kola 1+2 se obrací:** `univariate` (jen historie cíle) je teď LEPŠÍ než
  `multivariate`/`multivariate_plus` (+ exogenní featury) — dřív to bylo naopak. Bug dělal
  z čistě autoregresivního modelu uměle nefunkční model.
- Kolo 1 (`srovnani_1/`) i kolo 2 (`srovnani_2/`) kompletně přepočítány, staré výsledky
  zálohovány v `prediction/srovnani_1_prebugfix/` a `prediction/srovnani_2_prebugfix/`.
- `prediction_final/` přebudováno na `univariate` config (bez exogenních featur).

**Kolo 3 — HOTOVO** (`prediction/srovnani_3/`, detaily v `EXPERIMENT_LOG.md`
sekce "Kolo 3"). Uživatel požádal o další zlepšení nad 23.05 kW, autonomně
odzkoušeno přes `/loop`:
- `univariate_v2` (+ lag_2week/lag_3week) — vypadal slibně na 30d testu
  (21.36 kW), ale NEPROŠEL ověřením na 60d/staggered (23.30 kW, hůř než
  baseline) → zamítnuto, poučení: vždy ověřovat na 60d+staggered, ne jen 30d.
- **Blend 0.6×XGBoost + 0.4×SeasonalNaive24h** (shrinkage) — potvrzeno
  robustně na všech třech metodikách (30d: 21.13, 60d: 21.40, staggered:
  20.83 kW). Nasazeno do `prediction_final/`.
- Cestou nalezeny a opraveny 2 implementační chyby při nasazování blendu
  (kombinace hyperparameter-tuningu s alpha-search overfitovala; blend
  implementovaný jako single-model uvnitř sdíleného rekurzivního bufferu
  dával jiný, mnohem horší výsledek než dvě nezávislé rekurze) — obě
  zdokumentované v `EXPERIMENT_LOG.md` jako ponaučení.
- **`is_break` — budova je pracoviště VUT, spotřeba se řídí semestrem, ne
  jen dnem v týdnu** (upřesnění od uživatele). Přesný český kalendář svátků
  (`dateutil.easter`, ne odhad z paměti) + červenec/srpen jako hlavní
  prázdniny → výrazné, robustní zlepšení na všech třech metodikách (30d:
  -10.8 %, staggered: -6.0 %, 60d: -4.1 % samotná featura; v kombinaci s
  blendem 19.37/20.12/20.95 kW). Nasazeno do `prediction_final/`.

**Vítězná konfigurace k dnešnímu dni:** `univariate` + `is_break` +
blend(0.8×XGBoost + 0.2×SeasonalNaive24h) — 24h MAE = **19.37 kW** (30d
test), 20.12 kW staggered, 20.95 kW 60d. Uživatelův cíl (15-20 kW) dosažen
na 30d testu, na 60d/staggered těsně nad. Viz `prediction_final/runs/latest.json`
a `prediction_final/README.md` pro aktuální nasazenou verzi.

**Případné další kolo:**
- [ ] `srovnani_4+` — další feature nápady (blend s CatBoost, weekend/holiday featury)

**Finální výstup projektu:**
- [ ] Písemné doporučení: konkrétní feature set + model + hyperparametry pro produkční 24h/15min autoregresivní predikci, podložené čísly ze srovnání
- [x] Export finálního modelu (soubor) + reprodukovatelný trénovací skript → `prediction_final/train.py` + `runs/<run_id>/model.json`

**Budoucí fáze:**
- [x] **Python API pro predikci — HOTOVO** (`prediction_final/api/`, FastAPI — samostatná složka s vlastní kopií `common.py`, nezávislá na zbytku `prediction_final/`). Vstup = historická měření (JSON, UTC časy, min. 48h), výstup = 96 kroků (24h) predikce. Validace vstupu (formát/rozsah/mezery), `GET /health`. Ověřeno end-to-end (`api/test_api.py`): shoda s přímým výpočtem, "delší běh" přes 25 bodů v testovacím okně (MAE=18.85 kW, odpovídá dokumentovanému), 6 scénářů špatného vstupu správně odmítnuto. Detaily a 2 nalezené/opravené bugy v `EXPERIMENT_LOG.md` sekce "Python API pro predikci".
- [ ] Klientský skript na straně měřiče, který API volá — zatím jen `api_client_example.py` jako ukázka tvaru požadavku, ne skutečný meter-side klient

## Poznámky
- Soubor slouží jako sdílený zápisník úkolu — piš sem místo (nebo vedle) odpovědí v chatu.
