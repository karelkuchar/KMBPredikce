"""
Projde surova data ve slozce data/T12 P1.1/ (Main Archive CSV exporty,
1min interval), sjednoti je do jedne casove rady a resampluje na 15min krok.
Vystup: data_processed/main_archive_15min.csv

Pouziti:
    python3 data_processed/build_dataset.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "T12 P1.1"
OUT_PATH = Path(__file__).resolve().parent / "main_archive_15min.csv"

TIME_COL = "Record Time[s]"

# Sloupce ponechane ze surovych 1min Main Archive souboru.
# AVG_COLS se v ramci kazdeho 15min okna prumeruji, MIN_COLS/MAX_COLS berou
# min/max z jiz agregovanych 1min min/max sloupcu.
AVG_COLS = [
    "Avg.U1[V]", "Avg.U2[V]", "Avg.U3[V]",
    "Avg.I1[A]", "Avg.I2[A]", "Avg.I3[A]", "Avg.3I[A]",
    "Avg.3P[kW]", "Avg.P1[kW]", "Avg.P2[kW]", "Avg.P3[kW]",
    "Avg.3Q[kvar]", "Avg.Q1[kvar]", "Avg.Q2[kvar]", "Avg.Q3[kvar]",
    "Avg.3S[kVA]",
    "3PF[]",
    "Avg.f[Hz]",
    "Avg.THDU1[%]", "Avg.THDU2[%]", "Avg.THDU3[%]",
    "Avg.THDI1[%]", "Avg.THDI2[%]", "Avg.THDI3[%]",
]
MIN_COLS = ["Min.P1[kW]", "Min.P3[kW]", "Min.THDI1[%]"]
MAX_COLS = ["Max.P1[kW]", "Max.P3[kW]", "Max.THDI1[%]"]

USE_COLS = [TIME_COL] + AVG_COLS + MIN_COLS + MAX_COLS


def load_month(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=1,          # title radek
        header=0,
        encoding="utf-8-sig",
        usecols=lambda c: c in USE_COLS,
        low_memory=False,
    )
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="%d.%m.%Y %H:%M")
    return df.set_index(TIME_COL).sort_index()


def main() -> None:
    files = sorted(RAW_DIR.glob("*_Main Archive.csv"))
    if not files:
        raise SystemExit(f"Ve slozce {RAW_DIR} nebyly nalezeny zadne 'Main Archive' CSV soubory")

    print(f"Nalezeno {len(files)} Main Archive souboru")
    frames = []
    for f in files:
        df = load_month(f)
        frames.append(df)
        print(f"  {f.name}: {len(df)} radku, {df.index.min()} .. {df.index.max()}")

    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    print(f"\nSpojeno: {len(raw)} radku, {raw.index.min()} .. {raw.index.max()}")

    agg = {c: "mean" for c in AVG_COLS}
    agg.update({c: "min" for c in MIN_COLS})
    agg.update({c: "max" for c in MAX_COLS})

    resampled = raw.resample("15min").agg(agg)

    n_expected = len(
        pd.date_range(resampled.index.min(), resampled.index.max(), freq="15min")
    )
    n_missing = n_expected - resampled.dropna(how="all").shape[0]
    print(f"15min bloku: {len(resampled)} (ocekavano {n_expected}, "
          f"{n_missing} zcela prazdnych bloku -> mezery)")

    resampled.index.name = TIME_COL
    resampled.to_csv(OUT_PATH)
    print(f"\nUlozeno: {OUT_PATH} ({resampled.shape[0]} radku x {resampled.shape[1]} sloupcu)")

    gaps = resampled["Avg.3P[kW]"].isna()
    if gaps.any():
        gap_starts = resampled.index[gaps & ~gaps.shift(1, fill_value=False)]
        gap_ends = resampled.index[gaps & ~gaps.shift(-1, fill_value=False)]
        print(f"\n{len(gap_starts)} mezera(y) v cilove rade:")
        for s, e in list(zip(gap_starts, gap_ends))[:20]:
            print(f"  {s}  ..  {e}")
        if len(gap_starts) > 20:
            print(f"  ... a dalsich {len(gap_starts) - 20}")


if __name__ == "__main__":
    main()
