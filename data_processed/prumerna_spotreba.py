#!/usr/bin/env python3
"""
Spočítá průměrnou spotřebu z 15minutového archivu analyzátoru sítě.

Použití:
    python prumerna_spotreba.py                       # čte main_archive_15min.csv
    python prumerna_spotreba.py jiny_soubor.csv

Potřebuje: pip install pandas
"""
import sys
import pandas as pd

SOUBOR = sys.argv[1] if len(sys.argv) > 1 else "main_archive_15min.csv"
SLOUPEC_CAS = "Record Time[s]"
SLOUPEC_VYKON = "Avg.3P[kW]"

# Načtení – oddělovač (čárka, středník, tabulátor) se pozná automaticky
df = pd.read_csv(SOUBOR, sep=None, engine="python")
df.columns = df.columns.str.strip()

# Kdyby soubor používal desetinnou čárku, převede se na tečku
df[SLOUPEC_VYKON] = pd.to_numeric(
    df[SLOUPEC_VYKON].astype(str).str.replace(",", ".", regex=False),
    errors="coerce",
)
df[SLOUPEC_CAS] = pd.to_datetime(df[SLOUPEC_CAS], errors="coerce")
df = df.dropna(subset=[SLOUPEC_CAS, SLOUPEC_VYKON]).sort_values(SLOUPEC_CAS)

if df.empty:
    sys.exit("V souboru nejsou žádná platná data.")

# Délka intervalu v hodinách (odhad z nejčastějšího rozestupu záznamů, typicky 0,25 h)
interval_h = df[SLOUPEC_CAS].diff().mode()[0].total_seconds() / 3600
df["kWh"] = df[SLOUPEC_VYKON] * interval_h

zacatek = df[SLOUPEC_CAS].min()
konec = df[SLOUPEC_CAS].max()
pocet_hodin = len(df) * interval_h
celkem_kwh = df["kWh"].sum()
prum_kw = df[SLOUPEC_VYKON].mean()

print(f"Soubor:              {SOUBOR}")
print(f"Období:              {zacatek} – {konec}")
print(f"Počet záznamů:       {len(df)} (interval {interval_h * 60:.0f} min)")
print()
print(f"Průměrný výkon:      {prum_kw:.2f} kW")
print(f"Průměrně za hodinu:  {celkem_kwh / pocet_hodin:.2f} kWh")
print(f"Průměrně za den:     {celkem_kwh / pocet_hodin * 24:.1f} kWh")
print(f"Celková spotřeba:    {celkem_kwh:.1f} kWh")
print()
maxi = df.loc[df[SLOUPEC_VYKON].idxmax()]
mini = df.loc[df[SLOUPEC_VYKON].idxmin()]
print(f"Maximum (čtvrthod.): {maxi[SLOUPEC_VYKON]:.2f} kW  ({maxi[SLOUPEC_CAS]})")
print(f"Minimum (čtvrthod.): {mini[SLOUPEC_VYKON]:.2f} kW  ({mini[SLOUPEC_CAS]})")

# Spotřeba po dnech
denni = df.groupby(df[SLOUPEC_CAS].dt.date).agg(
    kWh=("kWh", "sum"),
    prum_kW=(SLOUPEC_VYKON, "mean"),
    max_kW=(SLOUPEC_VYKON, "max"),
    zaznamu=("kWh", "size"),
)
print("\nSpotřeba po dnech:")
print(f"{'Datum':<12}{'kWh':>10}{'prům. kW':>11}{'max kW':>10}{'záznamů':>10}")
for datum, r in denni.iterrows():
    print(f"{str(datum):<12}{r.kWh:>10.1f}{r.prum_kW:>11.2f}{r.max_kW:>10.2f}{r.zaznamu:>10.0f}")

nekompletni = denni[denni.zaznamu < round(24 / interval_h)]
if not nekompletni.empty:
    print("\nPozn.: některé dny nemají data za celých 24 h, jejich součet kWh je proto nižší.")
