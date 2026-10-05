"""
Test denniho behu BEZ zapisu do InfluxDB: nacte data z MS SQL, spocita
predikci presne stejne jako `db.py` (KROK "denne") a obojí ulozi do CSV.

    python3 dbtest.py

Potrebuje vedle sebe db.py, db_config.ini (staci sekce [mssql]) a model/.
Vystup (vedle tohoto souboru):
    dbtest_historie.csv   15min okna za poslednich DNI dni (vsechny sloupce)
    dbtest_predikce.csv   24h predikce Avg.3P[kW]
    dbtest_out.txt        vypis behu
Casy jsou v UTC i v mistnim case (sloupec cas_mistni).
"""

import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import db

DNI = 14   # kolik dni historie (pred nejnovejsim zaznamem v DB) ulozit do CSV; model sam pouzije db.PREDIKCE_HISTORIE_DNI

HIST_PATH = db.HERE / "dbtest_historie.csv"
PRED_PATH = db.HERE / "dbtest_predikce.csv"
OUT_PATH = db.HERE / "dbtest_out.txt"


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    tz = ZoneInfo(db.CAS_FEATUR)
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    to_ms = lambda dt: int((dt - epoch).total_seconds() * 1000)
    now = datetime.now(timezone.utc)

    with db.connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT measName FROM SmpMeasNameDB WHERE Id = ?", db.MEAS_ID)
        meter = cur.fetchone()[0]
        cur.execute("SELECT MAX(endTime) FROM UniArchiveBinPack WHERE keymeasName = ? AND keyArchID = ?",
                    db.MEAS_ID, db.ARCH_ID)
        newest = cur.fetchone()[0].replace(tzinfo=timezone.utc)
        out(f"=== dbtest {now:%Y-%m-%d %H:%M} UTC: {meter} (bez zapisu do Influxu) ===")
        out(f"  nejnovejsi zaznam v DB: {newest:%Y-%m-%d %H:%M} UTC = {newest.astimezone(tz):%d.%m. %H:%M} mistne"
            f" (stari dat {(now - newest).total_seconds() / 3600:.1f} h)")

        read_from = to_ms(newest) - max(DNI, db.PREDIKCE_HISTORIE_DNI) * 86400000
        windows, n_rec = db.aggregate_15min(db.read_records(cur, read_from, out))
    out(f"  nacteno {n_rec} 1min zaznamu -> {len(windows)} uplnych 15min oken")
    if not windows:
        out("  Zadna data.")
        return

    cols = list(db.AVG_COLS) + list(db.MIN_COLS) + list(db.MAX_COLS)
    with open(HIST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cas_utc", "cas_mistni"] + cols)
        for ms, row in windows.items():
            t = epoch.timestamp() + ms / 1000
            tu = datetime.fromtimestamp(t, timezone.utc)
            w.writerow([f"{tu:%Y-%m-%d %H:%M}", f"{tu.astimezone(tz):%Y-%m-%d %H:%M}"]
                       + [db.fmt_num(row.get(c)) for c in cols])
    first = datetime.fromtimestamp(epoch.timestamp() + min(windows) / 1000, timezone.utc)
    last = datetime.fromtimestamp(epoch.timestamp() + max(windows) / 1000, timezone.utc)
    out(f"  historie: {len(windows)} oken {first:%Y-%m-%d %H:%M} .. {last:%Y-%m-%d %H:%M} UTC -> {HIST_PATH.name}")

    pred_from = to_ms(newest) - db.PREDIKCE_HISTORIE_DNI * 86400000
    res = db.forecast_from_windows(windows, pred_from, out)
    if res is None:
        OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    times, preds, meta, tail = res
    t0 = tail.index[-1]
    with open(PRED_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cas_utc", "cas_mistni", "predikce_Avg.3P[kW]"])
        for t, v in zip(times, preds):
            w.writerow([f"{t:%Y-%m-%d %H:%M}", f"{t.tz_convert(db.CAS_FEATUR):%Y-%m-%d %H:%M}", repr(float(v))])
    out(f"  posledni skutecne okno (t0): {t0:%Y-%m-%d %H:%M} UTC = {t0.tz_convert(db.CAS_FEATUR):%d.%m. %H:%M} mistne, "
        f"{tail.iloc[-1]:.1f} kW; model dostal {len(tail)} kroku historie")
    out(f"  predikce (model {meta['run_id']}): {len(preds)} kroku "
        f"{times[0].tz_convert(db.CAS_FEATUR):%d.%m. %H:%M} .. {times[-1].tz_convert(db.CAS_FEATUR):%d.%m. %H:%M} mistne, "
        f"min {preds.min():.1f} / prumer {preds.mean():.1f} / max {preds.max():.1f} kW -> {PRED_PATH.name}")
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out(f"  vypis: {OUT_PATH.name}")


if __name__ == "__main__":
    main()
