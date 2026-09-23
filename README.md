# kmb_predikce

Predikce spotřeby elektřiny (`Avg.3P[kW]`) měřiče T12 P1.1, 24h dopředu po
15min krocích, pomocí jednoho rekurzivního modelu (XGBoost + SeasonalNaive24h
blend). Podrobný zápis zadání, rozhodnutí a experimentů: `UKOL.md` a
`prediction/EXPERIMENT_LOG.md`. Nasazený model + trénink/predikce/testy:
`prediction_final/` (`README.md` uvnitř). REST API: `prediction_final/api/`.

## Struktura

- `data/T12 P1.1/` — surová měsíční data z analyzátoru (Main Archive CSV, 1min).
- `data_processed/build_dataset.py` — sloučí surová data do
  `data_processed/main_archive_15min.csv` (15min krok).
- `prediction/` — experimenty a jejich zápis (`EXPERIMENT_LOG.md`).
- `prediction_final/` — finální, samostatný trénink/predikce/API (viz jeho README).

## Jak doplnit nová data a přetrénovat model

1. Zkopíruj nové měsíční CSV exporty do `data/T12 P1.1/` (stejné pojmenování
   jako stávající soubory).
2. `python3 data_processed/build_dataset.py` — přebuduje sloučený dataset.
3. `python3 prediction_final/train.py` — přetrénuje model, uloží nový běh do
   `prediction_final/runs/<datum_čas>/`, aktualizuje `runs/latest.json`.
4. `python3 prediction_final/test/run_test.py` — ověří přesnost na testovacím
   okně, porovnej `test_mae_24h_kW` s předchozím během.
5. Pokud běží `prediction_final/api/`, **restartuj ho** — model se načítá jen
   jednou při startu serveru.
