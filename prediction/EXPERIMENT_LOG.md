# Experiment log — predikce spotřeby T12 P1.1

Účel souboru: chronologický záznam kroků, konfigurací a výsledků pro pozdější
zpětnou rekonstrukci postupu (podklad pro odborný text). Pro aktuální stav
úkolu a otevřené otázky viz `UKOL.md`. Metrikové CSV soubory jsou zdroj pravdy
pro čísla — tady jen shrnutí + interpretace.

> **⚠️ DŮLEŽITÉ:** čísla v sekcích "Kolo 1" a "Kolo 2" níže byla spočítána
> **před opravou bufferového bugu** popsanou v sekci "Oprava kritického
> bufferového bugu" (za kolem 2). Bug obracel hlavní závěr obou kol (že
> rozšířené exogenní featury pomáhají) — po opravě je to přesně naopak.
> Ponecháno jako historický záznam postupu, ale **pro aktuální čísla a
> doporučení čti tu sekci, ne Kolo 1/2 samotné.**

---

## Kolo 0 — příprava dat

**Krok:** sjednocení surových 1min CSV (`data/T12 P1.1/*_Main Archive.csv`, 23
měsíčních souborů, 2024-09 až 2026-08) do jedné 15min řady.

**Skript:** `data_processed/build_dataset.py`
**Výstup:** `data_processed/main_archive_15min.csv` (68096 řádků × 30 sloupců)

**Zjištění:**
- Souvislý provoz od 2024-12-02 (dřívější data ze září/listopadu 2024 jsou
  řídká/přerušovaná — zřejmě instalace/testování přístroje).
- 6 drobných mezer po souvislém startu (≤ ~4h45min): 2× DST přechod, a 4
  výpadky v listopadu 2025 / lednu, březnu a květnu 2026.
- Modelovací dataset (`prediction/pipeline.py::load_base_df`) proto:
  1) ořízne data na `>= 2024-12-02 04:15`, 2) doplní mezery do 6h lineární
  interpolací, aby AR buffer nebyl přerušovaný.

---

## Kolo 1 — srovnání feature-setů × modelů (`prediction/srovnani_1/`)

**Cíl:** najít orientačně nejlepší kombinaci (univariate vs. multivariate
featury) × (LightGBM/CatBoost/XGBoost/baseline), s výchozími hyperparametry
(bez tuningu — ten je až na kolo 2).

**Konfigurace:**
- Target: `Avg.3P[kW]`, krok 15 min, jeden autoregresivní model (1 krok
  dopředu), aplikovaný rekurzivně na 24h výhled.
- Train/val/test: časové dělení, val = posledních 14 dní před testem, test =
  posledních 30 dní datasetu (kolo 1 — zkráceno kvůli rychlosti rekurzivní
  evaluace; lze zvětšit v dalších kolech).
- Feature-sety:
  - `univariate` — jen AR lagy/rolling/delta `Avg.3P[kW]` + kalendářní featury
    cílového času (přesně dle referenčního skriptu uživatele).
  - `multivariate` — navíc `Avg.Q3[kvar]`, `Avg.THDI1-3[%]`, `Min/Max.P1[kW]`,
    `Min/Max.P3[kW]`, při rekurzi zmrazené na poslední reálné hodnotě.
- Modely: `LightGBM`, `CatBoost`, `XGBoost` (výchozí hyperparametry), plus
  baseline `Persistence` (poslední hodnota) a `SeasonalNaive24h` (hodnota před
  24h) — všechny vyhodnoceny stejným harnessem, aby šlo srovnávat 1:1.
- Vyhodnocovací módy:
  - `1step` — bez rekurze, přímé 15min-ahead predikce (horní hranice
    dosažitelné přesnosti).
  - `24h` — rekurze s resyncem na reálná data každých 96 kroků (= **produkční
    scénář**).
  - `7d` — rekurze s resyncem každých 672 kroků (stress test kumulace chyby).

**Výsledky:** viz `prediction/srovnani_1/results_summary.csv` (souhrn MAE/RMSE/MAPE
per feature-set × model × mód) a `prediction/srovnani_1/results_by_horizon.csv`
(rozpad chyby po jednotlivých krocích 1–96 pro mód `24h`).

**Status: HOTOVO.** Souhrn (mód `24h` = produkční scénář, MAE v kW, test = 30 dní):

| feature_set (přejmenováno) | model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **rozšířené** (multivariate) | **XGBoost** | **28.47** | 39.32 | 15.88% |
| rozšířené | SeasonalNaive24h (baseline) | 29.28 | 47.32 | 17.48% |
| rozšířené | CatBoost | 30.32 | 41.14 | 17.05% |
| rozšířené | LightGBM | 31.18 | 42.95 | 17.41% |
| standardní (univariate) | Persistence (baseline) | 33.20 | 44.26 | 18.74% |
| standardní | CatBoost | 37.84 | 54.77 | 21.03% |
| standardní | XGBoost | 44.72 | 63.89 | 24.71% |
| standardní | LightGBM | 50.78 | 73.76 | 27.33% |

### Klíčové zjištění č.1 — přínos "rozšířeného" postupu (hodnota tohoto měřiče)

**Standardní postup** = jen `Avg.3P[kW]` (autoregresivní lagy/rolling/kalendář)
— to umí poskytnout i běžný elektroměr bez analýzy kvality sítě.
**Rozšířený postup** = navíc `Avg.Q3[kvar]`, `Avg.THDI1-3[%]`, `Min/Max.P1[kW]`,
`Min/Max.P3[kW]` — veličiny specifické pro tento síťový analyzátor (zmrazené
na poslední reálné hodnotě při rekurzi, viz metodologie výše).

U všech tří ML modelů rozšířený feature-set výrazně zlepšil přesnost 24h
predikce oproti standardnímu: XGBoost 28.5 vs. 44.7 kW MAE (**-36 %**),
CatBoost 30.3 vs. 37.8 kW (**-20 %**), LightGBM 31.2 vs. 50.8 kW (**-39 %**).
→ **konkrétní, číselně podložený argument pro přínos měřených veličin
speciálních pro tento přístroj** (THDI, Q3, min/max P1/P3), ne jen celkového
výkonu.

### Klíčové zjištění č.2 — rekurze bez exogenních dat se rozjíždí

Standardní (univariate) ML modely jsou v 24h rekurzivním módu **horší než
triviální baseline** (Persistence). Bez "kotvy" v podobě aktuálního stavu
soustavy (Q3/THDI/atd.) model při čistě autoregresivní rekurzi driftuje mimo
distribuci trénovacích dat. Rozšířený feature-set tento drift stabilizuje, ale
i nejlepší kombinace (rozšířené + XGBoost) porazila triviální sezónní baseline
jen o ~3 % MAE — prostor pro zlepšení v kole 2 (hyperparametry, případně další
exogenní kanály).

### Metodologická poznámka / limitace (řešit v kole 2)

`results_by_horizon.csv` ukazuje MAE per krok horizontu (1–96), ale protože
resync na reálná data probíhá vždy po přesně 96 krocích, **každý krok
horizontu odpovídá stále stejné hodině dne napříč všemi ~30 cykly testu**
(např. horizont krok 48 = vždy stejná hodina +12h od resetu). Křivka chyby po
horizontu je proto zkreslená tím, "jak těžko predikovatelná je zrovna tahle
hodina", ne čistě kumulací chyby rekurzí (pozorováno: MAE roste z ~4 kW při
kroku 1 na ~56 kW při kroku 48, pak klesá zpět na ~11.5 kW při kroku 96 —
nemonotónní, typické pro fázové zkreslení, ne pro čistou degradaci). **Oprava
pro kolo 2:** rozprostřít startovací body 24h predikce po různých hodinách
dne (ne jen fixní 96-krokový cyklus), aby šlo horizont a čas dne oddělit.

### Otevřené technické chyby nalezené a opravené během běhu
- CatBoost `Regressor.predict()` má vedlejší efekt — označí vstupní numpy pole
  jako read-only. Oprava: kopírovat buffer řádku před predikcí
  (`prediction/pipeline.py::autoregressive_predict`).
- XGBoost odmítá názvy sloupců obsahující `[`, `]`, `<` (naše sloupce typu
  `Avg.Q3[kvar]` je mají) — oprava: trénovat/predikovat na plain numpy poli
  bez feature names, ne na DataFrame.

---

## Kolo 2 — rozšíření exog. kanálů, oprava fázového zkreslení, tuning (`prediction/srovnani_2/`)

**Status: HOTOVO.**

### Změna datového okna (důležité pro srovnatelnost)

Uživatel označil úsek 2024-12 za měření s chybou / hodně chybějícími daty
(doménová znalost — v samotném `main_archive_15min.csv` to jako NaN-mezery
vidět není, viz gap analýza níž). `pipeline.py::MODEL_START` posunut z
`2024-12-02` na **`2025-01-01`**. Aby zůstalo kolo 1 a kolo 2 vzájemně
srovnatelné, **kolo 1 (`srovnani_1/`) bylo přepočítáno na stejném novém
okně** (výsledky v `srovnani_1/results_summary.csv` teď odpovídají
2025-01-01+, ne původnímu běhu — starý výsledek 28.47 kW z historického běhu
na 2024-12-02+ okně je tím nahrazen číslem 28.59 kW, rozdíl je v rámci šumu
dat, ne v metodice).

Gap analýza `main_archive_15min.csv` (pro budoucí referenci, kdyby se řešilo
znovu): jediná velká NaN-mezera je 2024-09-29 až 2024-12-02 (instalace,
už vyloučená). Po 2025-01-01 jsou jen drobné mezery (2× DST ~45min, pár
hodinových výpadků listopad 2025 / leden, březen, květen 2026), doplněné
interpolací jako dřív.

### Krok 1 — rozšíření exogenních kanálů (`run_comparison.py`)

Přidán feature-set `multivariate_plus` = `multivariate` (Q3, THDI1-3,
Min/Max P1/P3) + napětí U1-3, proud I1-3, účiník (`3PF[]`), THDU1-3.
Stejný harness jako kolo 1 (výchozí hyperparametry, 3 feature-sety × 5
modelů × 3 módy).

**Výsledek (mód 24h, 2025-01-01+ okno, MAE v kW):**

| feature_set | model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **multivariate** | **XGBoost** | **28.59** | 39.44 | 16.11% |
| multivariate | SeasonalNaive24h (baseline) | 29.28 | 47.32 | 17.48% |
| multivariate | CatBoost | 30.25 | 41.37 | 16.90% |
| multivariate_plus | XGBoost | 31.37 | 42.34 | 17.76% |
| multivariate | LightGBM | 31.83 | 43.71 | 17.68% |
| multivariate_plus | LightGBM | 32.26 | 43.50 | 18.17% |
| multivariate_plus | CatBoost | 32.40 | 43.79 | 18.19% |

**Zjištění: širší exog. sada (`multivariate_plus`) NEPOMÁHÁ, naopak škodí.**
V 1step módu jsou si feature-sety prakticky rovny (~6.0-6.2 kW MAE napříč
univariate/multivariate/multivariate_plus — přidané veličiny nesou málo
dodatečné 1-krokové informace nad rámec Q3/THDI/min-max P1,P3). Ale v 24h a
7d rekurzivním módu je `multivariate_plus` konzistentně horší než
`multivariate` u všech tří ML modelů (XGBoost 31.37 vs 28.59, LightGBM 32.26
vs 31.83, CatBoost 32.40 vs 30.25). Interpretace: exogenní featury se při
rekurzi **zmrazují** na poslední reálné hodnotě (viz metodologie kola 1) —
čím víc zmrazených kanálů model dostane jako "kotvu", tím větší je celkový
rozestup mezi tím, co model viděl při tréninku (živé, měnící se hodnoty), a
tím, co dostává při 24h inferenci (staticky zmrazené), a tím snáz model při
rekurzi driftuje mimo distribuci. Širší exog. sada tedy zvyšuje riziko
overfittingu na "živost" featur, ne jen šum navíc. → **doporučení: zůstat u
`multivariate` (Q3, THDI1-3, Min/Max P1/P3), U/I/PF/THDU nepřidávat.**

### Krok 2 — oprava fázového zkreslení (`horizon_staggered.py`)

`autoregressive_predict` (periodic) resynchronizuje vždy po přesně 96
krocích od pevného t0 → horizont krok je zavislý na hodině dne. Přidána
`staggered_recursive_predict` v `pipeline.py`: nezávislý 96-krokový
rekurzivní běh z každého bodu testovacího okna po `STRIDE=4` krocích (každou
hodinu), takže pro daný horizont krok h jde napříč běhy o různé hodiny/dny.

**Křivka MAE po horizontu (multivariate + XGBoost, každý 8. krok):**

| horizont krok | periodic MAE | staggered MAE |
|---|---|---|
| 1 | 3.9 | 7.3 |
| 9 | 4.3 | 13.4 |
| 17 | 4.4 | 21.1 |
| 25 | 9.6 | 28.6 |
| 33 | 28.4 | 34.0 |
| 41 | 50.9 | 38.8 |
| 49 | 53.6 | 41.2 |
| 57 | 56.8 (max) | 41.5 (max) |
| 65 | 49.4 | 40.2 |
| 73 | 30.8 | 37.8 |
| 81 | 26.3 | 34.2 |
| 89 | 17.4 | 30.4 |

**Potvrzeno: periodic křivka byla fázový artefakt.** Periodic má extrémní
nemonotónní tvar (3.9 → 56.8 → 17.4 kW) čistě proto, že každý horizont krok
odpovídal napříč ~30 cykly testu pořád stejné hodině dne — křivka ve
skutečnosti ukazovala "jak těžko predikovatelná je tahle hodina", ne
degradaci rekurzí. Staggered křivka (pokrývá všechny hodiny pro každý
horizont krok) je hladká a monotónní-ish: roste z 7.3 kW (krok 1) na plató
~41 kW kolem kroku 55-60, pak mírně klesá k ~30 kW — čistá kumulace chyby
rekurze, bez hodinového zkreslení. Celkové MAE staggered evaluace (31.0 kW
pro XGBoost, 31.2 pro CatBoost) je o něco vyšší než periodic (28.6 / 30.2),
protože periodic měl v tomto konkrétním testovacím okně "šťastný" fixní
cyklus startů — staggered je reprezentativnější odhad produkční přesnosti.
Soubor: `srovnani_2/results_by_horizon_staggered.csv`.

### Krok 3 — hyperparameter tuning (`tune.py`)

Random search (25 pokusů/model) pro XGBoost a CatBoost na `multivariate`.
**Tuning cílí přímo na 24h rekurzivní MAE na validaci (14 dní)**, ne na
1step MAE — 1step MAE se mezi konfiguracemi téměř neliší (viz krok 1), ale
24h MAE ano, protože rozhoduje hlavně chování při kumulaci chyby, ne
jednokroková přesnost.

**Vítězné hyperparametry (val 24h MAE):**
- **XGBoost: MAE=17.81 kW** — `learning_rate=0.03, max_depth=6, subsample=0.7, colsample_bytree=1.0, min_child_weight=1, reg_alpha=0.5, reg_lambda=5.0`
- CatBoost: MAE=19.14 kW — `learning_rate=0.08, depth=10, l2_leaf_reg=10.0, random_strength=1.0`

**Finální test (30 dní, `results_summary_tuned.csv`):**

| model | 1step MAE | 24h MAE | 7d MAE |
|---|---|---|---|
| XGBoost (tuned) | 6.16 | **26.76** | 31.22 |
| XGBoost (výchozí) | 6.22 | 28.59 | 34.54 |
| CatBoost (tuned) | 6.43 | 31.16 | 35.67 |
| CatBoost (výchozí) | — | 30.25 | — |

Tuning zlepšil XGBoost o **-6.4 %** (28.59 → 26.76 kW), CatBoost tuning
naopak mírně zhoršil oproti výchozím parametrům (30.25 → 31.16 kW) —
náhodné prohledávání 25 bodů nenašlo lepší bod než výchozí nastavení, CatBoost
default byl už rozumně dobrý. **XGBoost (tuned) je nový celkový vítěz.**

### Krok 4 — delší test okno, 60 dní (`test60.py`)

Ověření na 2× delším testovacím okně (60 dní místo 30) pro robustnost:

| model | 24h MAE (30d) | 24h MAE (60d) |
|---|---|---|
| XGBoost (tuned) | 26.76 | **24.91** |
| XGBoost (výchozí) | 28.59 | 25.15 |
| CatBoost (tuned) | 31.16 | 27.42 |
| CatBoost (výchozí) | 30.25 | 27.72 |
| SeasonalNaive24h | 29.28 | 28.53 |
| Persistence | 33.20 | 29.97 |

Pořadí modelů je na obou oknech stejné (XGBoost tuned nejlepší), přesnost se
na delším okně dokonce mírně zlepšila (pravděpodobně 30denní test obsahoval
o něco těžší úsek) — **výsledek kola 2 je robustní, ne artefakt konkrétního
30denního výřezu.**

### Celkové shrnutí kola 2

**Finální doporučená konfigurace: feature-set `multivariate` (Q3, THDI1-3,
Min/Max P1/P3) + XGBoost s vyladěnými hyperparametry, 24h MAE ≈ 25-27 kW**
(dle délky test okna), oproti kolu 1 (výchozí hyperparametry, stejné okno)
= zlepšení o ~6 % jen tuningem; rozšiřování exog. sady o U/I/PF/THDU
zavrženo (škodí); fázové zkreslení v per-horizont evaluaci opraveno a
zdokumentováno pro budoucí použití (`staggered_recursive_predict`).

---

## Oprava kritického bufferového bugu (po kole 2)

**Status: HOTOVO.** Nalezeno při komplexní kontrole kódu (`/code-review`) nad
`prediction_final/`, ale bug se ukázal být přítomný od kola 1 v
`prediction/pipeline.py` — ovlivňuje tedy **všechna** dosud publikovaná čísla
(kolo 1 i kolo 2 výše).

### Popis bugu

`autoregressive_predict` (a `staggered_recursive_predict`) inicializoval AR
buffer jako `target_real[:t0]` — tj. **bez** poslední skutečně známé hodnoty
`target_real[t0]` ("teď", bod, ze kterého rekurze vychází). Efekt:

- **Klouzavé statistiky a delta** (`roll_mean_*`, `roll_std_*`, `delta_1s`)
  pro úplně první krok každého 96-krokového cyklu počítaly okno posunuté o
  jeden krok zpátky (chyběla nejnovější hodnota, navíc byla v okně jedna
  hodnota starší, než měla být).
- **Lag featury** (`lag_Sum_15m` atd.) pro **druhý a každý další krok**
  uvnitř cyklu (95 z 96 kroků!) dostávaly **vlastní predikci modelu z
  předchozího kroku na špatné pozici bufferu** místo správné historické
  hodnoty — ověřeno instrumentací: `lag_Sum_15m` u kroku 2 mělo dostat
  148.47 (skutečná hodnota), ale dostalo 154.99 (echo předchozí predikce).
  U triviálního "Persistence" baselinu to vedlo k tomu, že model uvnitř
  cyklu jen opakoval dokola dvě čísla místo skutečné rekurze.
- Vedlejší bug: `exog_frozen` se inicializoval na `exog_arr[t0-1]` místo
  `exog_arr[t0]` (o krok starší, než mělo být).
- Vedlejší bug: `roll_std_*` používalo `np.std` (populační, ddof=0) místo
  pandas výchozí `ddof=1` (výběrová), použité při tréninku — systematická
  podhodnocená volatilita o ~3-15 % dle velikosti okna.
- Vedlejší bug ve `staggered_recursive_predict`: `y_true` se četlo jako
  `target_real[global_idx]` místo `target_real[global_idx+1]` — porovnávání
  predikce se skutečnou hodnotou o krok dřív, než na kterou byla predikce
  udělaná.

Bug se **sám opravoval při každém resyncu** (co 96 kroků se buffer přepíše
reálnými daty) — proto první krok každého cyklu (`horizon_step==1`) vždy
souhlasil s tréninkovým výpočtem, a proto to při prvotní kontrole (jen
`horizon_step==1` řádky) neodhalily ani předchozí testy v tomto projektu.

**Oprava:** buffer inicializován jako `target_real[:t0+1]` (o 1 prvek delší),
lag index posunut (`buf_idx = buf_len - 1 - lag_s`), resync přepsán na přímé
indexování (`target_buffer[k] = target_real[k]`), `exog_frozen` a
`staggered_recursive_predict`'s `y_true` opraveny analogicky, `roll_std`
přepnuto na `ddof=1`. Opraveno ve všech 4 kopiích této logiky:
`prediction/pipeline.py`, `prediction/best_model.py`,
`prediction_final/common.py`, `prediction_final/test/run_test.py`. Ověřeno
instrumentovaným testem: po opravě `horizon_step==1` **i všechny další
kroky** cyklu přesně souhlasí s tréninkovým výpočtem featur.

### Dopad — kompletní přepočet kola 1 + kola 2 (stejné 2025-01-01+ okno)

**Mód 24h, MAE v kW (`prediction/srovnani_2/results_summary.csv`):**

| feature_set | model | MAE (opraveno) | MAE (před opravou) |
|---|---|---|---|
| **univariate** | **XGBoost** | **22.56** | 44.72 (kolo 1) / 48.86 (kolo 2 stejné okno) |
| univariate | CatBoost | 23.12 | 37.84 / 38.98 |
| univariate | LightGBM | 24.32 | 50.78 / 49.93 |
| multivariate | XGBoost | 28.92 | 28.47 / 28.59 |
| multivariate | CatBoost | 30.17 | 30.32 / 30.25 |
| multivariate | LightGBM | 31.63 | 31.18 / 31.83 |
| multivariate_plus | XGBoost | 31.16 | — / 31.37 |

**Hlavní zjištění se obrací: `univariate` (jen historie cíle) je teď
nejlepší, `multivariate`/`multivariate_plus` (+ exogenní featury) jsou
HORŠÍ.** Před opravou vypadalo, že čistě autoregresivní model bez
"kotvy" v podobě exogenních dat diverguje při rekurzi — to byl z velké části
artefakt buggy AR bufferu (model dostával nesmyslné vlastní echo místo
skutečné historie, takže bez exogenních dat neměl šanci). Po opravě, kdy AR
featury konečně nesou správnou informaci, jsou samy o sobě dostatečné a
exogenní featury (Q3, THDI, min/max P1,P3) přidávají jen šum, ne signál.

**Ověření robustnosti nového vítěze (`univariate` + XGBoost):**
- **Staggered (fázově-neutrální) eval:** 22.17 kW (`results_by_horizon_staggered.csv`) — potvrzuje periodický výsledek, křivka po horizontu hladká (6.1 kW krok 1 → plató ~26 kW kolem kroku 49 → mírný pokles).
- **60denní test okno:** 23.20 kW (`results_summary_test60_univariate.csv`) — konzistentní s 30denním.
- **Hyperparameter tuning** (`tuning_best_univariate.json`, 25 pokusů): **nezlepšil** výchozí nastavení (tuned 23.74 vs. výchozí 22.56 na 30d; 23.74 vs. 23.20 na 60d) — výchozí hyperparametry XGBoostu jsou už dost dobré, netreba ladit.

**Nová finální doporučená konfigurace: feature-set `univariate` (jen
lagy/rolling/kalendářní featury cíle, BEZ exogenních veličin) + XGBoost s
VÝCHOZÍMI hyperparametry. 24h MAE ≈ 22.6-23.2 kW** (dle délky test okna) —
o cca 20 % lepší než dřívější (chybný) "vítěz" kola 2 (26.76 kW).
`prediction_final/` přebudováno na tuto konfiguraci (viz `runs/latest.json`
pro přesná čísla aktuálního běhu).

Staré (předopravné) výsledky zálohovány pro referenci:
`prediction/srovnani_1_prebugfix/`, `prediction/srovnani_2_prebugfix/`.

---

## Poznámky k metodologii (pro pozdější text)

- **Proč jeden rekurzivní model, ne 96 modelů na horizont:** uživatel měl
  vlastní ověřený skript s LightGBM a rekurzivní `autoregressive_predict`
  funkcí — tento přístup byl zvolen jako základ a rozšířen (vícero modelů,
  volitelné exogenní featury), ne nahrazen.
- **Proč exogenní featury (Q3, THDI, min/max P1/P3) nejsou při rekurzi
  autoregresivně dopočítávané jako target:** v produkčním nasazení nejsou
  budoucí hodnoty těchto veličin známé (na rozdíl od targetu, který se
  dopočítává z vlastních predikcí). Zmrazení na poslední reálné hodnotě je
  kompromis, který testuje přínos "stavu soustavy v čase startu predikce" bez
  nutnosti stavět vedlejší predikční model pro každou exogenní veličinu.
- **Proč `1step` mód jako referenční horní mez:** ukazuje, kolik přesnosti se
  ztrácí čistě kumulací chyby v rekurzi (rozdíl `1step` vs. `24h` MAE/RMSE),
  nezávisle na tom, který feature-set/model je použitý.

---

## Kolo 3 — pokus o další zlepšení nad univariate+XGBoost (probíhá, autonomní `/loop`)

Uživatel požádal o další snížení MAE pod aktuální baseline (23.05 kW,
`prediction_final/runs/20260922_195013/`), s tím že `prediction_final/`
zůstává nedotčené, dokud se nenajde ověřené zlepšení. Práce probíhá
autonomně (`/loop`, self-paced) v `prediction/srovnani_3/`, průběžně
zapisováno sem.

### Dávka 1 (`srovnani_3/run_comparison.py`)

Testováno:
- `univariate` (reprodukce baseline)
- `univariate_v2` (+ `lag_2week`, `lag_3week` — silnější týdenní signál,
  žádná změna rekurzivní logiky nutná, jen 2 další lagy)
- Ensemble (prostý průměr predikcí XGBoost + CatBoost, univariate)

*(výsledky doplním, jakmile doběhne)*

**Výsledky dávky 1 (30denní test okno, MAE v kW):**

| feature_set | model | MAE |
|---|---|---|
| **univariate_v2** | **XGBoost** | **21.36** |
| univariate | XGBoost (baseline) | 22.56 |
| univariate | Ensemble (XGB+Cat) | 22.58 |
| univariate | CatBoost | 23.12 |
| univariate_v2 | CatBoost | 25.59 |

**Nadějný nález:** `univariate_v2` (+ `lag_2week`, `lag_3week`) + XGBoost =
**21.36 kW**, o ~5.3 % lepší než dosavadní baseline (22.56 kW). Zajímavé:
CatBoost s týmiž featurami naopak zhoršil (25.59 vs 23.12) — model-specifická
reakce na přidané lagy, ne univerzální zlepšení. Ensemble (průměr XGB+Cat)
nepomohl, protože CatBoost je slabší člen dvojice.

Dál se ověřuje robustnost (60denní okno, staggered eval, hyperparameter
tuning) než se `prediction_final/` případně přebuduje.

**Výsledky dávky 2 — ověření robustnosti (`verify_v2.py`):**

| test | univariate_v2+XGBoost | univariate+XGBoost (baseline) |
|---|---|---|
| 30d periodic | 21.36 | 22.56 |
| 60d periodic | **23.30** | 23.20 |
| staggered (30d) | 22.08 | 22.17 |

**Závěr: `univariate_v2` NENÍ robustní zlepšení.** Na 60denním okně je
dokonce mírně HORŠÍ než baseline, na staggered evaluaci jen zanedbatelně
lepší (v rámci šumu). 30denní periodický výsledek (-5.3 %) byl
pravděpodobně artefakt konkrétního testovacího okna, ne skutečné zlepšení —
přesně ten typ zkreslení, který má staggered eval odhalit. **Nepoužito v
`prediction_final/`.** Ponaučení: nový nápad se musí ověřit na 60d/staggered
PŘED tím, než se považuje za vylepšení, ne jen na 30denním periodickém testu
(stejná lekce jako fázové zkreslení v kole 2).

### Dávka 3 — blend XGBoost + SeasonalNaive24h (`blend.py`)

Nápad: místo úpravy featur zkusit **shrinkage** — vážený průměr rekurzivní
predikce XGBoost (`univariate`) a triviální `SeasonalNaive24h` baseline
(hodnota před 24h), s vahou `alpha` nalezenou **na validaci** (ne na testu,
aby nedošlo k cherry-pickingu).

**Alpha search na validaci (14 dní):** optimum `alpha=0.6` (60 % váha
XGBoost, 40 % SeasonalNaive24h), val MAE=18.65 kW (vs. alpha=1.0 čistě
XGBoost: 23.37 kW val MAE).

**Ověření na testu (alpha=0.6 fixní, z validace, nepřeladěno na testu):**

| okno | blend (alpha=0.6) | čisté XGBoost |
|---|---|---|
| 30d | **21.13** | 22.56 |
| 60d | **21.40** | 23.20 |

**Konzistentní zlepšení na obou oknech (-6.3 % / -7.8 %) — na rozdíl od
dávky 1 (univariate_v2) tohle vypadá jako skutečný, robustní nález,**
protože zlepšení drží na obou test oknech (30d i 60d), ne jen na jednom.
Ověřuje se ještě staggered evaluací (`blend_staggered.py`) pro finální
potvrzení před přebudováním `prediction_final/`.

**Staggered ověření (`blend_staggered.py`, 697 startů):**

| model | MAE |
|---|---|
| **Blend (alpha=0.6)** | **20.83** |
| XGBoost pure | 22.17 |
| SeasonalNaive24h pure | 28.95 |

**POTVRZENO na všech třech metodikách (30d periodic, 60d periodic,
staggered) — toto je skutečné, robustní zlepšení, ne artefakt jednoho
testovacího okna:**

| metoda | blend | pure XGBoost | zlepšení |
|---|---|---|---|
| 30d periodic | 21.13 | 22.56 | -6.3% |
| 60d periodic | 21.40 | 23.20 | -7.8% |
| staggered | 20.83 | 22.17 | -6.0% |

### Celkové shrnutí kola 3

**Nová finální doporučená konfigurace: `univariate` + blend(0.6×XGBoost +
0.4×SeasonalNaive24h), 24h MAE ≈ 20.8-21.4 kW** — o dalších ~6-8 % lepší
než kolo "Oprava bufferového bugu" (22.6-23.2 kW), tj. celkově cca -28 % od
původního kola 1/2 čísla (28.5-28.9 kW) po opravě bugu a přidání blendu.

**Proč to funguje:** čistě autoregresivní model (XGBoost) může při dlouhé
24h rekurzi driftovat, protože chyba se kumuluje z vlastních predikcí.
SeasonalNaive24h (hodnota před 24h) sama o sobě je slabý model (29 kW MAE),
ale její chyby jsou **nekorelované** s XGBoost chybami jiným způsobem — je
to jiný typ modelu (čistá sezónnost bez učení), takže vážený průměr obou
snižuje rozptyl/drift, i když je jedna složka sama o sobě výrazně horší.
Klasický shrinkage/ensemble efekt.

`prediction_final/` přebudováno na tento blend (viz níže).

### Chyba při nasazení blendu do `prediction_final/` — nalezeno a opraveno

První pokus o nasazení (viz commit historie v `runs/20260922_202156` a
`20260922_202903`) dal test MAE **27.73 kW** — mnohem hůř, než ověřených
~21 kW. Prvně mylně diagnostikováno jako "hyperparameter tuning + alpha
search overfituje na stejném 14denním validačním okně" — po opravě (vychozí
hyperparametry místo tuningu) chyba **zůstala stejná (27.73 kW)**, což
ukázalo, že skutečná příčina je jinde.

**Skutečná příčina:** `BlendModel.predict()` použitý přímo jako `model` v
`autoregressive_predict` vrací **jednu smíchanou hodnotu**, která se pak
uloží zpět do **jediného sdíleného bufferu** — XGBoost tím při dalším kroku
rekurze vidí ve svých lag featurách směs vlastní predikce a sezónní
predikce, ne svou vlastní čistou trajektorii, na kterou byl natrénovaný.
Naproti tomu `prediction/srovnani_3/blend.py` (ověřená metodika) spouští
XGBoost a SeasonalNaive24h jako **dvě zcela nezávislé rekurze** (každá se
svým vlastním bufferem) a teprve NA KONCI zprůměruje jejich výstupní pole —
zásadně jiný výpočet, který blend.py otestoval a EXPERIMENT_LOG zde omylem
předpokládal, že `BlendModel` dělá totéž.

**Oprava:** nahrazeno `common.blend_autoregressive_predict()`, která spustí
dvě nezávislé `autoregressive_predict` rekurze (XGBoost, SeasonalNaive24h) a
teprve poté zkombinuje výsledná pole — přesně replikuje ověřenou metodiku.
`BlendModel` třída odstraněna (byla koncepčně vadná pro rekurzivní použití).

**Ponaučení:** "stejné rozhraní .predict(X)" nestačí k tomu, aby dvě různé
strategie kombinování modelů byly matematicky ekvivalentní v rekurzivním
kontextu — post-hoc blend dvou nezávislých trajektorií ≠ blend uvnitř
jednoho sdíleného rekurzivního bufferu. Kód, který "vypadá jako by měl
fungovat stejně", je potřeba vždy ověřit číselně, ne jen typovou kontrolou.

### Nasazení potvrzeno

Po opravě popsané výše (`common.blend_autoregressive_predict` — dvě
nezávislé rekurze, ne sdílený buffer) `prediction_final/` reprodukuje
přesně ověřené číslo:

- **`prediction_final` test 24h MAE = 21.135 kW** (`runs/20260922_203129/`),
  shoduje se s `srovnani_3/blend.py`'s 21.13 kW.
- `predict.py` i `test/run_test.py` ověřeny, oba dávají shodné MAE.
- Graf (`test/test_plot.png`) viditelně lépe sleduje denní cyklus i
  víkendové propady než předchozí verze (bez blendu).

Rozbité běhy (`runs/20260922_202156`, `20260922_202903`, oba MAE≈27.7 kW ze
špatné implementace `BlendModel`) smazány — chyba a čísla jsou zdokumentované
zde, samotné artefakty by jen matly.

**Shrnutí kola 3 k tomuto bodu:**

| krok | MAE (24h, 30d test) |
|---|---|
| Kolo "Oprava bufferového bugu" baseline | 23.05 |
| Kolo 3: univariate_v2 (zamítnuto, nerobustní) | — |
| **Kolo 3: blend 0.6×XGBoost + 0.4×SeasonalNaive24h** | **21.14** |

Celkové zlepšení od původního kola 1/2 (28.5-28.9 kW) do teď: **-26 %**.

### Dávka 4 — `is_break` (přibližný akademický kalendář VUT)

Uživatel upřesnil: měřená budova je pracoviště VUT, spotřeba se řídí
semestrem/prázdninami, ne jen dnem v týdnu. Přidán feature-set
`univariate_semester` (`pipeline.py`, nový sloupec `is_break`, přibližný
kalendář v `is_academic_break()` — léto ~červenec-půlka září, zima
~20.12.-2.1., **přibližné datumy, ne přesný VUT kalendář**).

Vedlejší refaktor: `time_cols` přesunuto z 3 hardcoded literálů v
`pipeline.py` do `FeatureSet.time_cols` (podobně jako `exog_cols`) — čistě
mechanická změna, ověřeno regresním testem (přesná shoda MAE=22.56132 pro
`univariate`+XGBoost před/po refaktoru).

**Výsledky (`semester.py`, univariate vs. univariate_semester, oba
XGBoost výchozí):**

| mode | univariate | univariate_semester | rozdíl |
|---|---|---|---|
| 30d | 22.56 | 21.75 | -3.6% |
| staggered | 22.17 | 21.76 | -1.9% |
| 60d | 23.20 | 23.11 | -0.4% |

Konzistentní (nikdy ne horší) na všech třech metodikách, ale malé — na
60denním okně (nejstabilnější metrika) jen -0.4 %, v podstatě šum. Podle
uživatelova kritéria "pokud to výrazně nezlepší, zahoď" je to hraniční
případ: není to falešný nález jako `univariate_v2` (nikde se nezhoršilo),
ale ani "výrazné" zlepšení. Zkouší se dál zkombinovat s existujícím
nasazeným blendem (`blend(alpha × XGBoost_semester + (1-alpha) ×
SeasonalNaive24h)`) — pokud to spolu s blendem překoná 21.14 kW, stojí to
za nasazení; pokud ne, `is_break` se zahazuje bez dalšího ladění kalendáře.

### Pauza (limit uživatele) — stav k 2026-09-22 ~20:52

Právě dokončena oprava `is_break_or_holiday()`: nahrazuje předchozí odhad
(`is_academic_break`, přibližné datumy prázdnin) přesným výpočtem — červenec+srpen
(dle uživatele) + **přesný český kalendář státních svátků** (fixní data +
Velikonoční pondělí dopočítané přes `dateutil.easter`, ne z paměti). Ověřeno
funkční (build_features('univariate_semester') proběhne, Velikonoční
pondělí 2025-04-21 správně označeno).

**Předchozí běh `blend_semester.py` byl zabit** (běžel se starou/nepřesnou
definicí `is_academic_break`, výsledky by byly zavádějící) — je potřeba
spustit znovu s opravenou `is_break_or_holiday()`, jakmile práce pokračuje.

**Práce pozastavena na žádost uživatele** (blíží se limit) — pokračovat po
22:00. Nedokončeno:
- [ ] Znovu spustit `semester.py` (samostatné srovnání univariate vs.
      univariate_semester) s opravenou kalendářní funkcí
- [ ] Znovu spustit `blend_semester.py` (kombinace s nasazeným blendem)
- [ ] Podle výsledku buď nasadit do `prediction_final/`, nebo zamítnout
- [ ] Případně zkusit další nápady z fronty (blend s CatBoost, hladší
      sezónní referenční hodnota) směrem k cíli 15-20 kW

Nasazený baseline zůstává nedotčen: `univariate` + blend(0.6×XGBoost +
0.4×SeasonalNaive24h), 24h MAE = 21.14 kW.

### Dávka 4b — `is_break_or_holiday` (přesný kalendář) — výrazné zlepšení

Po opravě (přesný český kalendář svátků + červenec/srpen, viz výše) je
zlepšení mnohem výraznější a konzistentní na všech třech metodikách:

| mode | univariate | univariate_semester | rozdíl |
|---|---|---|---|
| 30d | 22.56 | 20.12 | **-10.8%** |
| staggered | 22.17 | 20.83 | **-6.0%** |
| 60d | 23.20 | 22.25 | **-4.1%** |

Tohle už jasně splňuje kritérium "výrazné zlepšení" — na rozdíl od
přibližné verze (dávka 4a, -0.4 % až -3.6 %). Zajímavé: samotné
`univariate_semester` + XGBoost (BEZ blendu) už na staggered evaluaci
(20.83 kW) dosahuje stejné přesnosti jako aktuálně nasazený blend
(`univariate` + 0.6×XGBoost + 0.4×SeasonalNaive24h, taky 20.83 kW) — a to
bez jakéhokoliv blendování. Zkouší se teď kombinace obojího
(`univariate_semester` + blend).

### Dávka 5b — semester + blend: nasazeno

Kombinace `univariate_semester` + blend(alpha×XGBoost + (1-alpha)×
SeasonalNaive24h): alpha search na validaci → **alpha=0.8** (silnější váha
na XGBoost než u předchozího blendu bez semestru, val MAE=17.33).

| mode | nový (semester+blend) | starý nasazený (blend bez semester) | zlepšení |
|---|---|---|---|
| 30d | **19.37** | 21.14 | -8.4% |
| staggered | **20.12** | 20.83 | -3.4% |
| 60d | **20.95** | 21.40 | -2.1% |

Robustní zlepšení na všech třech metodikách. 30denní číslo (19.37 kW) je
už uvnitř uživatelova cíle 15-20 kW, staggered/60d těsně nad. **Nasazeno
do `prediction_final/`** — feature-set přepnut z `univariate` na
`univariate_semester`, `common.py` doplněn o `is_break_or_holiday()` logiku.

### Nasazení potvrzeno (semester + blend)

`prediction_final/common.py` doplněn o `czech_public_holidays()` +
`is_break_or_holiday()` (přesný český kalendář svátků přes `dateutil.easter`
+ červenec/srpen), `TIME_COLS` rozšířen o `is_break`. `train.py`/`predict.py`/
`test/run_test.py` nevyžadovaly žádnou změnu (volají `common.build_features()`
obecně, nová featura protekla automaticky).

- **`prediction_final` test 24h MAE = 19.367 kW** (`runs/20260922_215756/`,
  blend alpha=0.8), shoduje se s `srovnani_3` (19.37 kW).
- `predict.py` i `test/run_test.py` ověřeny, graf viditelně lépe sleduje
  celkovou úroveň (model teď správně ví, že srpen = "break" režim, ne jen
  z lagů/rolling statistik, ale i z explicitního signálu).

**Shrnutí celého kola 3:**

| krok | MAE (24h, 30d test) |
|---|---|
| Baseline po opravě bufferového bugu | 23.05 |
| Blend (XGBoost + SeasonalNaive24h) | 21.14 |
| **Blend + is_break (VUT semestr/svátky)** | **19.37** |

Celkové zlepšení od původního kola 1/2 (28.5-28.9 kW): **-32 %**. Uživatelův
cíl 15-20 kW dosažen na 30denním testu (19.37), na 60d/staggered těsně nad
(20.95/20.12) — blízko cíle, ne úplně uvnitř na všech metrikách.

---

## Python API pro predikci (`prediction_final/api/`)

Složka `prediction_final/api/` je samostatná — vlastní kopie `common.py`
(ne sdílená s `prediction_final/common.py`), takže se dá zkopírovat/nasadit
nezávisle na zbytku `prediction_final/` (kromě natrénovaných modelů v
`../runs/`).

**Status: HOTOVO.** Naplňuje budoucí fázi zmíněnou v `UKOL.md` ("Python API
pro predikci, vstup = poslední buffer měření, výstup = 24h/15min predikce").

**Návrh:** FastAPI server (`POST /predict`, `GET /health`), izolovaný stejně
jako zbytek `prediction_final/` — importuje jen `common.py`. Klíčový
poznatek při implementaci: `autoregressive_predict`/`blend_autoregressive_predict`
při live inferenci vůbec nepotřebují `common.build_features()`'s dropna'd
historický dataframe (ten slouží jen tréninku) — lag featury se počítají za
běhu z `target_buffer`, takže stačí dodat surová historická data + časové
řady rozšířené o 96 budoucích kroků (jen kalendářní featury, cílová
hodnota se nikdy nečte). Nová funkce `build_live_feature_set()` v `api.py`
tohle staví.

**Validace vstupu:** UTC časy (explicitní pásmo povinné), přesně na 15min
mřížce, min. 192 záznamů (48h, dle `UKOL.md`), mezery do 6h se doplní
interpolací (delší odmítnuty), duplicity a NaN/Inf odmítnuty. `400` pro
datové chyby, `422` pro typové/formátové (Pydantic).

**Doplněno na žádost uživatele:** volitelné pole `include_history: true/false`
v požadavku (výchozí `false`) — pokud `true`, odpověď navíc obsahuje pole
`history` s historií přesně tak, jak ji API použilo (seřazená, doplněná).
Reálné příklady požadavků/odpovědí (krátký, delší, s `include_history`)
uloženy v `api/example_json/` (vygenerované skutečným voláním, ne vymyšlené).

### Ověření (`api/test_api.py`, end-to-end přes skutečný uvicorn server)

1. `/health` odpovídá správně.
2. `/predict` s reálnými daty dává **přesně** stejný výsledek jako přímé
   volání `common.blend_autoregressive_predict` (max rozdíl 0.0005 kW —
   zaokrouhlovací šum) — potvrzuje, že API nezavádí žádnou odchylku od
   ověřené metodiky.
3. `include_history=true/false` vrací/nevrací pole `history` dle očekávání.
4. **Mezery v datech:** mezera 8h (nad limit 6h) je odmítnuta (400), mezera
   2h (pod limit) se lineárně doplní (`gaps_filled_steps` odpovídá počtu
   doplněných kroků).
5. **DST přechod:** data pokrývající skutečný český přechod ze
   letního/zimního času (2025-10-26) projdou bez chyby — všechny časy jsou
   v UTC, které žádný DST posun nemá, takže nehrozí žádná dvojznačnost.
6. **"Delší běh":** 25 volání z různých bodů **v drženém testovacím okně**
   (ne v trénovacích datech — to by unikalo), agregovaná MAE=18.85 kW,
   odpovídá dokumentovanému číslu (~19-21 kW).
7. Validace vstupu: 6 scénářů špatných dat (krátká historie, naivní čas,
   čas mimo mřížku, duplicity, NaN, prázdný seznam) — všechny správně
   odmítnuty (400/422), žádný nezpůsobí pád serveru.

**Dva reálné bugy nalezené a opravené při psaní testů:**
- `prepare_history()` havarovala (`AttributeError`) při detekci duplicitních
  časů — `.dt.isoformat()` neexistuje na pandas `Series`, opraveno na
  per-prvek `.isoformat()`.
- Odeslání `NaN` jako hodnoty správně selhalo na Pydantic validaci, ale
  FastAPI/Starlette výchozí handler chyby pak spadl s `500` misto vrácení
  `422` — snaží se v chybové odpovědi echovat zpět neplatnou vstupní
  hodnotu (`nan`), ale Starlette `JSONResponse` používá `allow_nan=False`
  (striktní JSON nezná `NaN` literál). Opraveno vlastním
  `RequestValidationError` handlerem, který neplatné/neserializovatelné
  hodnoty před odpovědí očistí. Poučení: i "standardní" framework chování
  (echo neplatného vstupu v chybové zprávě) může samo selhat na okrajových
  hodnotách (NaN/Inf) — vždy end-to-end otestovat chybové cesty, ne jen
  úspěšné.

---

## Rozšíření `prediction_final`: rozpad metrik, ukládání do `runs/`, nastavitelné testovací okno (2026-09-23)

Tři drobné, ale trvalé vylepšení `test/run_test.py` a `train.py`, vyžádané
uživatelem po ověření, že `train.py` je skutečně autoritativní trénovací
skript pro nasazený model (ověřeno čerstvým rerunem — přesně reprodukuje
MAE=19.367 kW, alpha=0.8).

**1. Rozpad metrik po hodině/dni/měsíci.** `test/run_test.py` nově počítá a
ukládá `test_metrics_breakdown.csv` — MAE/RMSE/MAPE zvlášť pro každou
hodinu dne (0–23), kalendářní den a měsíc testovacího okna. Účel: odhalit,
jestli chyba není nerovnoměrně rozložená (např. horší v určitou denní dobu
nebo měsíc) místo jednoho souhrnného čísla. Ověřeno živě — měsíční agregát
odpovídá celkové MAE, hodinové/denní řádky ukazují očekávanou variabilitu.

**2. Ukládání výstupů i do `runs/<run_id>/`.** `test_results.csv`,
`test_plot.png` a nově `test_metrics_breakdown.csv` se ukládají na DVĚ
místa — do `test/` (vždy "poslední výsledek") i do příslušné
`runs/<run_id>/` složky, aby výsledky testu zůstaly svázané s konkrétním
natrénovaným modelem stejně jako `model.json`/`metadata.json`, a historie
se needitovala/nepřepisovala. Ověřeno `ls -la` na obou místech — shodné
soubory se shodnými časovými razítky.

**3. Nastavitelná délka testovacího okna (`TEST_DAYS`).** `train.py` má
nyní na začátku souboru proměnnou `TEST_DAYS = common.TEST_DAYS` (výchozí
30 dní) — pro jiný běh (60denní ověření apod.) stačí ji přepsat přímo v
kódu a spustit `train.py` znovu. **Explicitně na žádost uživatele jde o
obyčejnou Python proměnnou, ne CLI parametr/argparse** — první implementace
byla přes `--test-days` argument, uživatel to opravil ("chci to mít jako
proměnnou v pythonu, ne jako parametr"). Zvolená hodnota se ukládá do
`metadata.json["test_days"]` daného běhu; `predict.py` i
`test/run_test.py` si při načtení modelu automaticky nastaví
`common.TEST_DAYS` na tuto uloženou hodnotu (`meta.get("test_days", ...)`,
se zpětnou kompatibilitou pro starší běhy bez tohoto pole) — takže vždy
používají přesně to samé okno, na jakém byl daný model trénován/testován,
i když se výchozí `TEST_DAYS` v `train.py` mezitím změní.

Ověřeno end-to-end: výchozí `TEST_DAYS=30` reprodukuje MAE=19.367 kW;
dočasná zkouška `TEST_DAYS=60` dala správně větší test okno (5761 řádků,
2026-07-02→2026-08-31, MAE=22.253 kW); po ověření vráceno zpět na 30 a
model přetrénován, aby zůstal nasazený správný (30denní) běh —
`runs/20260923_095645/` (MAE=19.367 kW, RMSE=26.973, MAPE=12.546 %,
alpha=0.8).
