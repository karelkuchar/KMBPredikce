"""
Kolo 2, krok 2: oprava fazoveho zkresleni v per-horizont evaluaci.

Kolo 1 (`results_by_horizon.csv`) resynchronizuje vzdy po presne 96 krocich od
pevneho t0 -> horizont krok je zavisly na hodine dne (kazdy krok horizontu
odpovida stale stejne hodine napric vsemi cykly testu). Tento skript spousti
navic "staggered" evaluaci: nezavisly 96-krokovy rekurzivni beh z mnoha
ruznych startovnich bodu rozprostrenych po testovacim okne (krok STRIDE), takze
pro dany horizont krok h jde napric behy o ruzne hodiny/dny.

Srovna obe krivky (periodic vs. staggered) pro vitezne kombinace z kola 1/2,
aby bylo videt, kolik z puvodni nemonotonni krivky byl fazovy artefakt.

Vystup:
    prediction/srovnani_2/results_by_horizon_staggered.csv

Pouziti:
    python3 prediction/srovnani_2/horizon_staggered.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as pl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
STRIDE = 4  # kazda hodina (4 kroky x 15min) - pokryje vsechny hodiny dne napric testem

# Kombinace k overeni - vitez po opravene bufferove logice (univariate) +
# puvodni kolo 1/2 vitez (multivariate) pro srovnani.
COMBOS = [
    ("univariate", "XGBoost"),
    ("multivariate", "XGBoost"),
    ("multivariate", "CatBoost"),
]


def horizon_curve(df_run, tag):
    rows = []
    for h, g in df_run.groupby("horizon_step"):
        m = pl.evaluate(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        rows.append(dict(horizon_step=int(h), horizon_min=int(h) * pl.RESAMPLE_MIN,
                          n=len(g), eval_type=tag, **m))
    return rows


def main():
    t_start = time.time()
    all_rows = []

    for fs_mode, model_name in COMBOS:
        print(f"\n{'='*70}\n{fs_mode} + {model_name}\n{'='*70}")
        fs = pl.build_features(fs_mode)
        train, val, test = pl.split_train_val_test(fs.df)
        t0 = len(fs.df) - len(test)
        n_total = len(fs.df)

        trainer = {**pl.BASELINE_TRAINERS, **pl.MODEL_TRAINERS}[model_name]
        predict_fn = trainer(train, val, fs.feature_cols)

        # periodic (kolo 1 zpusob) - pro primou srovnatelnost prepocitano stejnym
        # kodem/modelem (ne jen nacteno z kola 1, protoze featury/RNG se mohly lisit)
        t0m = time.time()
        preds_p, hstep_p = pl.autoregressive_predict(predict_fn, fs, n_test=len(test), reset_steps=pl.RESET_24H)
        y_true_p = fs.df["TARGET"].iloc[t0:].to_numpy(np.float32)
        df_periodic = pd.DataFrame(dict(horizon_step=hstep_p, y_true=y_true_p, y_pred=preds_p))
        rows_p = horizon_curve(df_periodic, "periodic")
        for r in rows_p:
            r["feature_set"] = fs_mode
            r["model"] = model_name
        all_rows.extend(rows_p)
        print(f"   periodic  done ({time.time()-t0m:.1f}s), MAE overall="
              f"{pl.evaluate(y_true_p, preds_p)['mae']}")

        # staggered (fix) - start kazdych STRIDE kroku, horizon=96, nezavisle rekurze
        t0m = time.time()
        start_indices = list(range(t0, n_total - 96 + 1, STRIDE))
        df_stag = pl.staggered_recursive_predict(predict_fn, fs, start_indices, horizon=96)
        rows_s = horizon_curve(df_stag, "staggered")
        for r in rows_s:
            r["feature_set"] = fs_mode
            r["model"] = model_name
        all_rows.extend(rows_s)
        overall_stag = pl.evaluate(df_stag["y_true"].to_numpy(), df_stag["y_pred"].to_numpy())
        print(f"   staggered done ({time.time()-t0m:.1f}s), {len(start_indices)} startu, "
              f"MAE overall={overall_stag['mae']}")

    df_out = pd.DataFrame(all_rows).sort_values(["feature_set", "model", "eval_type", "horizon_step"])
    df_out.to_csv(OUT_DIR / "results_by_horizon_staggered.csv", index=False)
    print(f"\nUlozeno: {OUT_DIR / 'results_by_horizon_staggered.csv'}")
    print(f"Celkovy cas: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
