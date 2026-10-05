"""
Pripojeni k MS SQL na PC-EMVUT (FreeTDS + pyodbc, Windows/NTLM login).

Na VM je potreba vedle sebe:
    db.py           - tento kod
    db_config.ini   - prihlasovaci udaje (MS SQL + InfluxDB; pokud chybi, db.py vytvori sablonu)
    model/          - natrenovany model (model.json, metadata.json) - jen pro KROK "denne"

Spusteni:
    python3 db.py                # provede KROK (promenna nize)
    KROK=export python3 db.py    # jednorazove jiny krok bez upravy souboru

Vystup se ulozi i do db_out.txt.

Instalace a nastaveni VM: README.md
"""

import configparser
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "db_config.ini"
OUT_PATH = HERE / "db_out.txt"
LOG_PATH = HERE / "db_freetds.log"

DEBUG = False   # True = zapise podrobny FreeTDS log do db_freetds.log (pri problemech s pripojenim)

# co se ma udelat:
#   "denne"   - (spousti cron 1x denne) nova data z MS SQL -> 15min historie + 24h predikce -> InfluxDB
#   "export"  - vytahne data mista MEAS_ID z binarniho archivu do CSV (15min, format pro model)
#   "prehled" - tabulky, sloupce, ukazkove radky cele databaze
#   "archiv"  - rozbor binarniho archivu jednoho mericiho mista (MEAS_ID)
KROK = os.environ.get("KROK") or "denne"
MEAS_ID = 3     # SmpMeasNameDB.Id - 3 = T12 P1.1

# denne
MODEL_DIR = HERE / "model" if (HERE / "model").exists() else HERE.parent / "model"
OD_HISTORIE = None          # prvni beh (prazdny bucket): "2025-01-01" = historie od tohoto dne (UTC), None = vse z DB
PREPSAT_DNI = 1             # kazdy beh znovu zapise poslednich N dni historie (doplni okna, ktera byla minule neuplna)
PREDIKCE_HISTORIE_DNI = 8   # kolik dni historie dostane model (lag_1week potrebuje 7 dni)
MIN_HISTORIE_KROKU = 192    # min. 48h souvisle historie, jinak se predikce nedela (stejne jako API)
MAX_MEZERA_KROKU = 24       # mezery do 6h se interpoluji (stejne jako pri treninku)
HORIZONT_KROKU = 96         # 24h dopredu po 15 min
# model byl trenovan na mistnim case (puvodni CSV exporty) - kalendarni featury (hodina, den v tydnu,
# svatky) se proto pocitaji v tomto pasmu, historie i predikce se ale ukladaji v UTC
CAS_FEATUR = "Europe/Prague"
INFLUX_BATCH = 5000         # radku line protocolu na jeden zapis

# export
ARCH_ID = 0             # 0 = hlavni 1min archiv
OD = None               # "2025-01-01" = jen data od tohoto dne (UTC), None = vse
ULOZIT_1MIN = False     # True = ulozit i 1min data (velky soubor, stovky MB)
CHUNK_RECORDS = 5000    # kolik zaznamu se stahuje jednim dotazem
TIMEZONE = "Europe/Prague"  # cas ve vystupnim CSV (stejne jako puvodni CSV exporty)

# sloupec vystupniho CSV -> (PropName v archivu, nasobitel); W -> kW, var -> kvar, VA -> kVA
AVG_COLS = {
    "Avg.U1[V]": ("U_avg_U1", 1), "Avg.U2[V]": ("U_avg_U2", 1), "Avg.U3[V]": ("U_avg_U3", 1),
    "Avg.I1[A]": ("I_avg_I1", 1), "Avg.I2[A]": ("I_avg_I2", 1), "Avg.I3[A]": ("I_avg_I3", 1),
    "Avg.3I[A]": ("I_avg_3I", 1),
    "Avg.3P[kW]": ("P_avg_3P", 1e-3), "Avg.P1[kW]": ("P_avg_P1", 1e-3),
    "Avg.P2[kW]": ("P_avg_P2", 1e-3), "Avg.P3[kW]": ("P_avg_P3", 1e-3),
    "Avg.3Q[kvar]": ("Q_avg_3Q", 1e-3), "Avg.Q1[kvar]": ("Q_avg_Q1", 1e-3),
    "Avg.Q2[kvar]": ("Q_avg_Q2", 1e-3), "Avg.Q3[kvar]": ("Q_avg_Q3", 1e-3),
    "Avg.3S[kVA]": ("S_avg_3S", 1e-3),
    "3PF[]": None,  # dopocitava se: |P_avg_3P| / S_avg_3S
    "Avg.f[Hz]": ("f_avg_f", 1),
    "Avg.THDU1[%]": ("Harmonics/THDU_avg_THDU1", 1), "Avg.THDU2[%]": ("Harmonics/THDU_avg_THDU2", 1),
    "Avg.THDU3[%]": ("Harmonics/THDU_avg_THDU3", 1),
    "Avg.THDI1[%]": ("Harmonics/THDI_avg_THDI1", 1), "Avg.THDI2[%]": ("Harmonics/THDI_avg_THDI2", 1),
    "Avg.THDI3[%]": ("Harmonics/THDI_avg_THDI3", 1),
}
MIN_COLS = {
    "Min.P1[kW]": ("P_min_P1", 1e-3), "Min.P3[kW]": ("P_min_P3", 1e-3),
    "Min.THDI1[%]": ("Harmonics/THDI_min_THDI1", 1),
}
MAX_COLS = {
    "Max.P1[kW]": ("P_max_P1", 1e-3), "Max.P3[kW]": ("P_max_P3", 1e-3),
    "Max.THDI1[%]": ("Harmonics/THDI_max_THDI1", 1),
}

# pruzkum databaze: z tabulek, jejichz nazev obsahuje neco z KEYWORDS, se vypisou ukazkove radky
KEYWORDS = ["archiv", "meas", "device", "point", "channel", "quantity"]
SAMPLE_ROWS = 5
MAX_TEXT = 80   # zkraceni dlouhych textu ve vypisu
HEX_BYTES = 32  # kolik bajtu binarnich dat ukazat

CONFIG_TEMPLATE = """\
[mssql]
; jmeno sekce v /etc/freetds/freetds.conf
servername = ws11
database = VUT25
driver = FreeTDS
; STROJ\\uzivatel (lokalni ucet) nebo DOMENA\\uzivatel, bez uvozovek
uid = PC-EMVUT\\EM VUT
pwd =

[influx]
url = http://localhost:8086
org =
token =
bucket_history = history
bucket_forecast = forecast
"""

FREETDS_CONF = Path("/etc/freetds/freetds.conf")
FREETDS_SECTION_CMD = (
    "printf '\\n[ws11]\\n\\thost = 147.229.159.14\\n\\tport = 1433\\n"
    "\\ttds version = 7.4\\n\\tencryption = off\\n' | sudo tee -a /etc/freetds/freetds.conf"
)


# ---------------------------------------------------------------- pripojeni

def load_config(section="mssql", keys=("servername", "database", "driver", "uid", "pwd")):
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        raise SystemExit(f"Vytvoren {CONFIG_PATH} - vypln v nem pwd (pripadne uid) a spust znovu.")
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(CONFIG_PATH, encoding="utf-8")
    if section not in cp:
        tmpl = CONFIG_TEMPLATE[CONFIG_TEMPLATE.index(f"[{section}]"):].split("\n\n[")[0]
        with open(CONFIG_PATH, "a", encoding="utf-8") as f:
            f.write("\n" + tmpl.rstrip() + "\n")
        raise SystemExit(f"Do {CONFIG_PATH} jsem doplnil sekci [{section}] - vypln v ni prazdne hodnoty a spust znovu.")
    cfg = dict(cp[section])
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        raise SystemExit(f"V {CONFIG_PATH} v sekci [{section}] chybi hodnoty: {', '.join(missing)}")
    return cfg


def connect(timeout=10):
    cfg = load_config()

    if FREETDS_CONF.exists() and f"[{cfg['servername']}]" not in FREETDS_CONF.read_text(errors="replace"):
        raise SystemExit(
            f"V {FREETDS_CONF} chybi sekce [{cfg['servername']}]. Pridej ji:\n  {FREETDS_SECTION_CMD}"
        )

    if DEBUG:
        os.environ["TDSDUMP"] = str(LOG_PATH)  # musi byt pred nactenim ovladace
    import pyodbc

    # UID/PWD zabaleny do {} - bezpecne i pri mezere/zpetnem lomitku/strednicich
    conn_str = (
        f"DRIVER={{{cfg['driver']}}};"
        f"SERVERNAME={cfg['servername']};"
        f"DATABASE={cfg['database']};"
        f"UID={{{cfg['uid']}}};"
        f"PWD={{{cfg['pwd']}}};"
    )
    return pyodbc.connect(conn_str, timeout=timeout)


# ---------------------------------------------------------------- pruzkum

def fmt(v):
    if v is None:
        return "NULL"
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        return f"<{len(b)} B: {b[:HEX_BYTES].hex(' ')}{' ...' if len(b) > HEX_BYTES else ''}>"
    s = str(v)
    return s if len(s) <= MAX_TEXT else s[:MAX_TEXT] + "..."


def explore(cur, out):
    # 1) tabulky + pocty radku (sys.partitions nevyzaduje zvlastni opravneni)
    cur.execute("""
        SELECT s.name, t.name, SUM(p.rows)
        FROM sys.tables t
        JOIN sys.schemas s ON s.schema_id = t.schema_id
        JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
        GROUP BY s.name, t.name
        ORDER BY s.name, t.name
    """)
    tables = cur.fetchall()
    out(f"\n=== 1) Tabulky ({len(tables)}) ===")
    for schema, name, rows in tables:
        out(f"  {schema}.{name:<45} {rows:>14,} radku")

    cur.execute("""
        SELECT s.name, v.name FROM sys.views v
        JOIN sys.schemas s ON s.schema_id = v.schema_id ORDER BY s.name, v.name
    """)
    views = cur.fetchall()
    if views:
        out(f"\n--- Pohledy (views) ({len(views)}) ---")
        for schema, name in views:
            out(f"  {schema}.{name}")

    # 2) sloupce
    cur.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE,
               CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """)
    out("\n=== 2) Sloupce ===")
    current = None
    for schema, table, col, dtype, maxlen, nullable in cur.fetchall():
        if (schema, table) != current:
            current = (schema, table)
            out(f"\n  [{schema}.{table}]")
        length = "" if maxlen is None else f"({'max' if maxlen == -1 else maxlen})"
        out(f"    {col:<40} {dtype}{length}{'' if nullable == 'YES' else '  NOT NULL'}")

    # 3) ukazkove radky z kandidatnich tabulek
    candidates = [(s, n) for s, n, _ in tables if any(k in n.lower() for k in KEYWORDS)]
    out(f"\n=== 3) Ukazkove radky (tabulky obsahujici {KEYWORDS}) ===")
    if not candidates:
        out("  (zadna tabulka neodpovida - uprav KEYWORDS podle seznamu vyse)")
    for schema, name in candidates:
        out(f"\n  [{schema}.{name}]")
        try:
            cur.execute(f"SELECT TOP {SAMPLE_ROWS} * FROM [{schema}].[{name}]")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            if not rows:
                out("    (prazdna)")
            for i, row in enumerate(rows, 1):
                out(f"    -- radek {i}")
                for c, v in zip(cols, row):
                    out(f"      {c:<38} {fmt(v)}")
        except Exception as e:
            out(f"    CHYBA: {e}")


def hexdump(b, out, indent="      "):
    for i in range(0, len(b), 32):
        out(f"{indent}{i:5d}: {b[i:i + 32].hex(' ')}")


def inspect_archive(cur, out):
    import struct
    from datetime import datetime, timedelta

    epoch = datetime(2000, 1, 1)  # cas v archivu = ms od 2000-01-01 (UTC)

    cur.execute("SELECT Id, identifyDB, measName FROM SmpMeasNameDB ORDER BY Id")
    out("\n=== Merici mista (SmpMeasNameDB) ===")
    for mid, ident, name in cur.fetchall():
        out(f"  {mid:>4}  identify={ident!s:>4}  {name}{'   <<<' if mid == MEAS_ID else ''}")

    cur.execute("""
        SELECT keyArchID, ArchDef, Period, COUNT(*), SUM(Count), MIN(keyTime), MAX(endTime),
               MIN(DATALENGTH(Data) / NULLIF(Count, 0)), MAX(DATALENGTH(Data) / NULLIF(Count, 0)),
               SUM(DATALENGTH(DataBuf))
        FROM UniArchiveBinPack WHERE keymeasName = ?
        GROUP BY keyArchID, ArchDef, Period ORDER BY keyArchID
    """, MEAS_ID)
    groups = cur.fetchall()
    out(f"\n=== Archivy mista {MEAS_ID} (UniArchiveBinPack) ===")
    out("  archID archDef  period[ms]  baliku    zaznamu  od                       do                       B/zaznam   DataBuf[B]")
    for a, d, per, n, cnt, t0, t1, lmin, lmax, buf in groups:
        out(f"  {a:>6} {d!s:>7} {per!s:>11} {n:>7} {cnt!s:>10}  {t0!s:<24} {t1!s:<24} {lmin}-{lmax}   {buf}")

    cur.execute("""
        SELECT keyArchID, keyTime, endTime, Count, DATALENGTH(Data), DATALENGTH(DataBuf), TimeIndexes
        FROM UniArchiveBinPack WHERE keymeasName = ? ORDER BY keyArchID, keyTime
    """, MEAS_ID)
    out("\n=== Baliky (TimeIndexes = dvojice [index 4B][cas 8B]) ===")
    for a, t0, t1, cnt, ldata, lbuf, ti in cur.fetchall():
        ti = bytes(ti or b"")
        pairs = []
        for i in range(0, len(ti) - 11, 12):
            idx, ms = struct.unpack(">IQ", ti[i:i + 12])
            pairs.append(f"{idx}@{epoch + timedelta(milliseconds=ms):%Y-%m-%d %H:%M:%S}")
        out(f"  arch={a} {t0} .. {t1}  count={cnt}  data={ldata} buf={lbuf}  ti={len(ti)}B: {', '.join(pairs[:6])}"
            f"{' ...' if len(pairs) > 6 else ''}")

    for a, d, *_ in groups:
        cur.execute("SELECT TypeXML FROM UniArchiveDefinition WHERE Id = ?", d)
        row = cur.fetchone()
        out(f"\n=== Definice archivu ArchDef={d} (archID {a}) ===")
        out(str(row[0]) if row else "  (nenalezena)")

        # prvni 2 zaznamy nejstarsiho baliku
        cur.execute("""
            SELECT TOP 1 DATALENGTH(Data) / NULLIF(Count, 0),
                   SUBSTRING(Data, 1, 2 * (DATALENGTH(Data) / NULLIF(Count, 0)))
            FROM UniArchiveBinPack WHERE keymeasName = ? AND keyArchID = ? AND Count > 0
            ORDER BY keyTime
        """, MEAS_ID, a)
        row = cur.fetchone()
        if not row or not row[0]:
            continue
        reclen, raw = int(row[0]), bytes(row[1])
        out(f"\n  -- prvni 2 zaznamy (archID {a}, {reclen} B/zaznam), hex:")
        hexdump(raw, out)
        ms = struct.unpack(">Q", raw[:8])[0]
        n = (reclen - 8) // 4
        floats = struct.unpack(f">{n}f", raw[8:8 + 4 * n])
        out(f"  -- zaznam 1: cas {epoch + timedelta(milliseconds=ms)} UTC, zbytek jako float32 BE "
            f"({n} hodnot, {reclen - 8 - 4 * n} B navic):")
        for i in range(0, n, 8):
            out("      " + "  ".join(f"{i + j:3d}:{v:11.4f}" for j, v in enumerate(floats[i:i + 8])))


# ---------------------------------------------------------------- export

XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"
TYPE_SIZES = {
    "System.Single": 4, "System.Double": 8, "System.Byte": 1, "System.SByte": 1,
    "System.Int16": 2, "System.UInt16": 2, "System.Int32": 4, "System.UInt32": 4,
    "System.Int64": 8, "System.UInt64": 8, "System.DateTime": 8,
}


def parse_layout(type_xml):
    """Z UniArchiveDefinition.TypeXML spocita offset kazdeho pole v zaznamu.

    Zaznam = 8 B cas (ms od 2000-01-01 UTC, big-endian) + pole typu UniPropDesc
    v poradi podle XML; VirtualPropDesc se jen dopocitavaji a v datech nejsou.
    Velikost pole urcuje TypeSer (<len> u skalaru NEsedi - napr. Reset UInt16
    ma len=4, ale v datech 2 B); u poli (TypeSer konci "[]") je <len> pocet
    prvku, u System.String pocet bajtu.
    Vraci ({PropName: (offset, TypeSer, big_endian)}, delka zaznamu).
    """
    import xml.etree.ElementTree as ET

    fields, off = {}, 8
    for p in ET.fromstring(type_xml).iter("PropDesc"):
        if p.get(XSI_TYPE) != "UniPropDesc":
            continue
        typ, n = p.findtext("TypeSer"), int(p.findtext("len"))
        base = typ[:-2] if typ.endswith("[]") else typ
        if base == "System.String":
            size = n
        elif base in TYPE_SIZES:
            size = TYPE_SIZES[base] * (n if typ.endswith("[]") else 1)
        else:
            raise ValueError(f"Neznamy typ {typ} ({p.findtext('PropName')})")
        fields[p.findtext("PropName")] = (off, typ, p.findtext("BigEndian") == "true")
        off += size
    return fields, off


def make_decoder(fields):
    """Funkce (buf, offset zaznamu) -> (cas UTC v ms, {sloupec: hodnota})."""
    import struct

    specs = []
    for col, src in {**AVG_COLS, **MIN_COLS, **MAX_COLS}.items():
        if src is None:
            continue
        prop, mul = src
        if prop not in fields:
            raise KeyError(f"V definici archivu chybi pole {prop} (pro sloupec {col})")
        off, typ, be = fields[prop]
        if typ != "System.Single":
            raise TypeError(f"{prop}: ocekavan System.Single, je {typ}")
        specs.append((col, off, ">f" if be else "<f", mul))

    def decode(buf, rec):
        ms = struct.unpack_from(">Q", buf, rec)[0]
        vals = {col: struct.unpack_from(fmt, buf, rec + off)[0] * mul for col, off, fmt, mul in specs}
        p, s = vals["Avg.3P[kW]"], vals["Avg.3S[kVA]"]
        vals["3PF[]"] = abs(p) / s if s else float("nan")
        return ms, vals

    return decode


def fmt_num(v):
    return "" if v is None or v != v else repr(v)


def export(cur, out):
    import csv
    import math
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    tz = ZoneInfo(TIMEZONE)
    od_ms = None if OD is None else int((datetime.fromisoformat(OD).replace(tzinfo=timezone.utc) - epoch)
                                        .total_seconds() * 1000)

    cur.execute("SET TEXTSIZE 2147483647")  # jinak muze server oriznout varbinary(max)
    cur.execute("SELECT measName FROM SmpMeasNameDB WHERE Id = ?", MEAS_ID)
    row = cur.fetchone()
    meas_name = row[0] if row else f"meas{MEAS_ID}"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in meas_name)
    out(f"\n=== Export: {meas_name} (Id {MEAS_ID}), archID {ARCH_ID} ===")

    cur.execute("""
        SELECT CAST(StreamId AS varchar(36)), keyTime, endTime, ArchDef, Count,
               DATALENGTH(Data), DATALENGTH(DataBuf)
        FROM UniArchiveBinPack WHERE keymeasName = ? AND keyArchID = ? ORDER BY keyTime
    """, MEAS_ID, ARCH_ID)
    packs = cur.fetchall()

    layouts = {}
    all_cols = list(AVG_COLS) + list(MIN_COLS) + list(MAX_COLS)
    windows = {}        # 15min okno (mistni cas, naive) -> [soucty, pocty, min, max]
    last_local = None   # pro preskoceni duplicitni hodiny pri prechodu z letniho casu
    n_rec = n_dup = n_skip_od = 0
    preview = []

    f1 = w1 = None
    if ULOZIT_1MIN:
        f1 = open(HERE / f"{safe_name}_1min.csv", "w", newline="", encoding="utf-8")
        w1 = csv.writer(f1)
        w1.writerow(["Record Time[s]", "time_utc"] + all_cols)

    for stream_id, t0, t1, arch_def, count, ldata, lbuf in packs:
        if od_ms is not None and t1 is not None and t1 < datetime.fromisoformat(OD):
            continue
        if arch_def not in layouts:
            cur.execute("SELECT TypeXML FROM UniArchiveDefinition WHERE Id = ?", arch_def)
            fields, reclen = parse_layout(str(cur.fetchone()[0]))
            layouts[arch_def] = (make_decoder(fields), reclen)
            offs = ", ".join(f"{AVG_COLS[c][0]}@{fields[AVG_COLS[c][0]][0]}"
                             for c in ("Avg.3P[kW]", "Avg.3S[kVA]", "Avg.THDI1[%]"))
            out(f"  ArchDef {arch_def}: {reclen} B/zaznam ({offs})")
        decode, reclen = layouts[arch_def]

        ok = (ldata or 0) == int(count) * reclen
        out(f"  balik {t0} .. {t1}: {int(count)} zaznamu, ArchDef {arch_def}"
            f"{'' if ok else f'  !! DATALENGTH {ldata} != {int(count)}*{reclen}'}"
            f"{f', DataBuf {lbuf} B' if lbuf else ''}")

        for col, length in (("Data", ldata or 0), ("DataBuf", lbuf or 0)):
            if length % reclen:
                out(f"    !! {col}: {length} B neni nasobek {reclen}, preskakuji")
                continue
            chunk = CHUNK_RECORDS * reclen
            for start in range(0, length, chunk):
                n = min(chunk, length - start)
                cur.execute(f"SELECT SUBSTRING({col}, ?, ?) FROM UniArchiveBinPack "
                            "WHERE StreamId = CAST(? AS uniqueidentifier)", start + 1, n, stream_id)
                buf = bytes(cur.fetchone()[0])
                if len(buf) != n:
                    raise RuntimeError(f"Server vratil {len(buf)} B misto {n} B (oriznuti?)")
                for rec in range(0, n, reclen):
                    ms, vals = decode(buf, rec)
                    if od_ms is not None and ms < od_ms:
                        n_skip_od += 1
                        continue
                    utc = epoch + timedelta(milliseconds=ms // 60000 * 60000)
                    local = utc.astimezone(tz).replace(tzinfo=None)
                    if last_local is not None and local <= last_local:
                        n_dup += 1   # stejne jako puvodni build_dataset.py: duplicity -> keep first
                        continue
                    last_local = local
                    n_rec += 1
                    if len(preview) < 3:
                        preview.append((local, vals))
                    if w1:
                        w1.writerow([local.strftime("%Y-%m-%d %H:%M:%S"), utc.strftime("%Y-%m-%d %H:%M:%S")]
                                    + [fmt_num(vals[c]) for c in all_cols])

                    key = local.replace(minute=local.minute // 15 * 15, second=0, microsecond=0)
                    acc = windows.get(key)
                    if acc is None:
                        acc = windows[key] = [{c: 0.0 for c in AVG_COLS}, {c: 0 for c in AVG_COLS},
                                              {c: math.inf for c in MIN_COLS}, {c: -math.inf for c in MAX_COLS}]
                    for c in AVG_COLS:
                        v = vals[c]
                        if v == v:
                            acc[0][c] += v
                            acc[1][c] += 1
                    for c in MIN_COLS:
                        if vals[c] == vals[c]:
                            acc[2][c] = min(acc[2][c], vals[c])
                    for c in MAX_COLS:
                        if vals[c] == vals[c]:
                            acc[3][c] = max(acc[3][c], vals[c])
    if f1:
        f1.close()

    out(f"\n  1min zaznamu: {n_rec}  (preskoceno: {n_dup} duplicit/DST, {n_skip_od} pred OD)")
    for local, vals in preview:
        out(f"    {local}  Avg.3P={vals['Avg.3P[kW]']:.3f} kW  Avg.U1={vals['Avg.U1[V]']:.3f} V  "
            f"3PF={vals['3PF[]']:.3f}  THDI1={vals['Avg.THDI1[%]']:.3f} %")
    if not windows:
        out("  Zadna data.")
        return

    path15 = HERE / f"{safe_name}_15min.csv"
    t, t_end = min(windows), max(windows)
    n_win = n_empty = 0
    with open(path15, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Record Time[s]"] + all_cols)
        while t <= t_end:
            acc = windows.get(t)
            if acc is None:
                n_empty += 1
                w.writerow([t.strftime("%Y-%m-%d %H:%M:%S")] + [""] * len(all_cols))
            else:
                row = [acc[0][c] / acc[1][c] if acc[1][c] else None for c in AVG_COLS]
                row += [acc[2][c] if acc[2][c] != math.inf else None for c in MIN_COLS]
                row += [acc[3][c] if acc[3][c] != -math.inf else None for c in MAX_COLS]
                w.writerow([t.strftime("%Y-%m-%d %H:%M:%S")] + [fmt_num(v) for v in row])
            n_win += 1
            t += timedelta(minutes=15)
    out(f"  15min oken: {n_win} ({min(windows)} .. {t_end}), z toho {n_empty} prazdnych (mezery)")
    out(f"  Ulozeno: {path15}")
    if ULOZIT_1MIN:
        out(f"  Ulozeno: {HERE / f'{safe_name}_1min.csv'}")


# ---------------------------------------------------------------- cteni archivu (pro "denne")

def get_layout(cur, arch_def, cache):
    if arch_def not in cache:
        cur.execute("SELECT TypeXML FROM UniArchiveDefinition WHERE Id = ?", arch_def)
        fields, reclen = parse_layout(str(cur.fetchone()[0]))
        cache[arch_def] = (make_decoder(fields), reclen)
    return cache[arch_def]


def first_index_from(cur, stream_id, col, n, reclen, from_ms):
    """Binarni hledani prvniho zaznamu s casem >= from_ms (zaznamy v baliku jsou serazene)."""
    import struct

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        cur.execute(f"SELECT SUBSTRING({col}, ?, 8) FROM UniArchiveBinPack "
                    "WHERE StreamId = CAST(? AS uniqueidentifier)", mid * reclen + 1, stream_id)
        if struct.unpack(">Q", bytes(cur.fetchone()[0]))[0] < from_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


def read_records(cur, from_ms, out):
    """Generator 1min zaznamu mista MEAS_ID (archiv ARCH_ID) s casem >= from_ms:
    (ms od 2000-01-01 UTC, {sloupec: hodnota}), po balicich v casovem poradi.
    Nic nedrzi v pameti - pri prvnim behu jde o celou historii (~1 mil. zaznamu)."""
    from datetime import datetime, timedelta

    from_dt = datetime(2000, 1, 1) + timedelta(milliseconds=from_ms)
    cur.execute("SET TEXTSIZE 2147483647")  # jinak muze server oriznout varbinary(max)
    cur.execute("""
        SELECT CAST(StreamId AS varchar(36)), keyTime, endTime, ArchDef, DATALENGTH(Data), DATALENGTH(DataBuf)
        FROM UniArchiveBinPack
        WHERE keymeasName = ? AND keyArchID = ? AND (endTime IS NULL OR endTime >= ?)
        ORDER BY keyTime
    """, MEAS_ID, ARCH_ID, from_dt)
    packs = cur.fetchall()

    layouts = {}
    for stream_id, t0, t1, arch_def, ldata, lbuf in packs:
        decode, reclen = get_layout(cur, arch_def, layouts)
        for col, length in (("Data", ldata or 0), ("DataBuf", lbuf or 0)):
            if length % reclen:
                out(f"  !! balik {t0}: {col} {length} B neni nasobek {reclen} B, preskakuji")
                continue
            n = length // reclen
            start = first_index_from(cur, stream_id, col, n, reclen, from_ms)
            for i in range(start, n, CHUNK_RECORDS):
                k = min(CHUNK_RECORDS, n - i)
                cur.execute(f"SELECT SUBSTRING({col}, ?, ?) FROM UniArchiveBinPack "
                            "WHERE StreamId = CAST(? AS uniqueidentifier)", i * reclen + 1, k * reclen, stream_id)
                buf = bytes(cur.fetchone()[0])
                if len(buf) != k * reclen:
                    raise RuntimeError(f"Server vratil {len(buf)} B misto {k * reclen} B (oriznuti?)")
                for r in range(0, len(buf), reclen):
                    yield decode(buf, r)


def aggregate_15min(records):
    """1min zaznamy -> 15min okna v UTC (stejne jako export: Avg = prumer, Min/Max = min/max).
    Vraci ({zacatek okna v ms od 2000-01-01: {sloupec: hodnota}} jen pro UPLNE uplynula okna,
    pocet zaznamu)."""
    acc, n, last_ms = {}, 0, None
    for ms, vals in records:
        n += 1
        last_ms = ms if last_ms is None else max(last_ms, ms)
        w = ms // 900000 * 900000
        a = acc.setdefault(w, ({}, {}, {}, {}))
        for c in AVG_COLS:
            v = vals[c]
            if v == v:
                a[0][c] = a[0].get(c, 0.0) + v
                a[1][c] = a[1].get(c, 0) + 1
        for c in MIN_COLS:
            if vals[c] == vals[c]:
                a[2][c] = min(a[2].get(c, vals[c]), vals[c])
        for c in MAX_COLS:
            if vals[c] == vals[c]:
                a[3][c] = max(a[3].get(c, vals[c]), vals[c])
    if not n:
        return {}, 0
    last_minute_end = last_ms // 60000 * 60000 + 60000
    out = {}
    for w, (sums, cnts, mins, maxs) in sorted(acc.items()):
        if w + 900000 > last_minute_end:
            continue  # posledni okno jeste neni cele - zapise se pri dalsim behu
        row = {c: sums[c] / cnts[c] for c in sums}
        row.update(mins)
        row.update(maxs)
        out[w] = row
    return out, n


# ---------------------------------------------------------------- InfluxDB (HTTP API v2, bez knihoven)

def lp_escape(s, extra=""):
    for ch in "\\," + extra + " ":
        s = s.replace(ch, "\\" + ch)
    return s


def influx_request(cfg, path, params, body, headers):
    import urllib.error
    import urllib.parse
    import urllib.request

    url = cfg["url"].rstrip("/") + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Authorization": f"Token {cfg['token']}", **headers})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"InfluxDB {path}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}")


def influx_write(cfg, bucket, lines):
    for i in range(0, len(lines), INFLUX_BATCH):
        influx_request(cfg, "/api/v2/write", {"org": cfg["org"], "bucket": bucket, "precision": "s"},
                       "\n".join(lines[i:i + INFLUX_BATCH]).encode("utf-8"),
                       {"Content-Type": "text/plain; charset=utf-8"})


def influx_last_time(cfg, bucket, measurement, meter, field):
    """Cas posledniho bodu (datetime UTC) nebo None, kdyz bucket zatim nic nema."""
    import csv
    import io
    import json
    from datetime import datetime

    flux = (f'from(bucket: "{bucket}") |> range(start: 0) '
            f'|> filter(fn: (r) => r._measurement == "{measurement}" and r.meter == "{meter}" and r._field == "{field}") '
            f'|> last() |> keep(columns: ["_time"])')
    text = influx_request(cfg, "/api/v2/query", {"org": cfg["org"]},
                          json.dumps({"query": flux, "type": "flux"}).encode("utf-8"),
                          {"Content-Type": "application/json", "Accept": "application/csv"})
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if len(rows) < 2 or "_time" not in rows[0]:
        return None
    return datetime.fromisoformat(rows[1][rows[0].index("_time")].replace("Z", "+00:00"))


def lines_history(windows, meter, from_ms):
    tag = f"mereni,meter={lp_escape(meter, '=')}"
    lines = []
    for w, row in windows.items():
        if w < from_ms:
            continue
        fields = ",".join(f"{lp_escape(c, '=')}={v!r}" for c, v in row.items() if v == v and abs(v) != float("inf"))
        if fields:
            lines.append(f"{tag} {fields} {(w // 1000) + EPOCH_2000_S}")
    return lines


EPOCH_2000_S = 946684800  # 2000-01-01T00:00:00Z v unixovem case


# ---------------------------------------------------------------- predikce (bez API, model primo zde)

def czech_holidays(years):
    from datetime import date, timedelta
    from dateutil.easter import easter

    fixed = [(1, 1), (5, 1), (5, 8), (7, 5), (7, 6), (9, 28), (10, 28), (11, 17), (12, 24), (12, 25), (12, 26)]
    days = {date(y, m, d) for y in years for m, d in fixed}
    days |= {easter(y) + timedelta(days=1) for y in years}  # Velikonocni pondeli
    return days


def time_features(times_local):
    """Kalendarni featury - 1:1 jako prediction_final/common.py (time_feats + is_break)."""
    import numpy as np

    # float32 jako v common.py - XGBoost je na hranach prahu citlivy i na rozdil float32/float64
    t = times_local
    hour, minute, dow = t.hour.to_numpy(), t.minute.to_numpy(), t.dayofweek.to_numpy()
    slot = (hour * 4 + minute // 15).astype(np.float32)
    month, doy = t.month.to_numpy().astype(np.float32), t.dayofyear.to_numpy().astype(np.float32)
    holidays = czech_holidays(range(t.year.min(), t.year.max() + 1))
    is_break = np.isin(t.month, [7, 8]) | np.array([d in holidays for d in t.date])
    return {
        "hour": hour, "minute": minute, "dayofweek": dow,
        "day_sin": np.sin(2 * np.pi * slot / 96), "day_cos": np.cos(2 * np.pi * slot / 96),
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
        "year_sin": np.sin(2 * np.pi * doy / 365.25), "year_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_weekend": (dow >= 5).astype(int), "is_break": is_break.astype(int),
    }


def predict_24h(times_utc, values):
    """times_utc: souvisla 15min rada (pandas DatetimeIndex v UTC), values: Avg.3P[kW] bez NaN.
    Vraci (casy predikce UTC, hodnoty) - 96 kroku. Stejny vypocet jako
    prediction_final/api (blend alpha*XGBoost + (1-alpha)*SeasonalNaive24h, kazdy
    jako vlastni rekurze), jen kalendarni featury v CAS_FEATUR (jako pri treninku)."""
    import json

    import numpy as np
    import pandas as pd
    from xgboost import XGBRegressor

    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    model = XGBRegressor()
    model.load_model(str(MODEL_DIR / "model.json"))
    cols, lag_map = meta["feature_cols"], meta["lag_map"]
    col_idx = {c: k for k, c in enumerate(cols)}

    n = len(values)
    step = pd.Timedelta(minutes=15)
    future = pd.date_range(times_utc[-1] + step, periods=HORIZONT_KROKU, freq="15min")
    # featury radku = cas cile (radek + 15 min), v mistnim case jako pri treninku
    target_times = (times_utc[-1:].append(future) + step).tz_convert(CAS_FEATUR).tz_localize(None)
    tf = time_features(target_times)

    def run(predict_one):
        buf = [float(v) for v in values]
        row = np.zeros((1, len(cols)), dtype=np.float32)
        preds = []
        for i in range(HORIZONT_KROKU):
            for col, lag in lag_map.items():
                j = len(buf) - 1 - lag
                row[0, col_idx[col]] = buf[j] if j >= 0 else buf[0]
            for w in (4, 16, 32):
                arr = buf[-w:]
                row[0, col_idx[f"roll_mean_{w}s"]] = np.mean(arr)
                row[0, col_idx[f"roll_std_{w}s"]] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            row[0, col_idx["delta_1s"]] = buf[-1] - buf[-2]
            for c in meta["time_cols"]:
                row[0, col_idx[c]] = tf[c][i]
            p = float(np.clip(predict_one(row), -1e6, 1e6))
            preds.append(p)
            buf.append(p)
        return np.array(preds)

    p_xgb = run(lambda r: model.predict(r)[0])
    p_naive = run(lambda r: r[0, col_idx["lag_1day"]])
    alpha = meta["blend_alpha"]
    assert n >= 2
    return future, alpha * p_xgb + (1 - alpha) * p_naive, meta


def forecast_from_windows(windows, pred_from, out):
    """15min okna (z aggregate_15min) -> 24h predikce z oken >= pred_from.
    Vraci (casy UTC, hodnoty, metadata modelu, pouzita historie) nebo None."""
    import numpy as np
    import pandas as pd

    epoch = pd.Timestamp("2000-01-01", tz="UTC")
    ser = pd.Series({epoch + pd.Timedelta(milliseconds=w): r.get("Avg.3P[kW]", np.nan)
                     for w, r in windows.items() if w >= pred_from}, dtype=float)
    if ser.dropna().empty:
        out(f"  !! predikce se NEDELA: za poslednich {PREDIKCE_HISTORIE_DNI} dni pred nejnovejsim zaznamem nejsou data")
        return None
    ser = ser.reindex(pd.date_range(ser.index[0], ser.index[-1], freq="15min"))
    ser = ser.interpolate(limit=MAX_MEZERA_KROKU, limit_area="inside")
    gaps = ser.isna().to_numpy()
    tail = ser.iloc[np.where(gaps)[0][-1] + 1:] if gaps.any() else ser  # souvisly konec bez delsich mezer
    if len(tail) < MIN_HISTORIE_KROKU:
        out(f"  !! predikce se NEDELA: souvisle historie jen {len(tail)} kroku "
            f"(potreba {MIN_HISTORIE_KROKU} = 48h, delsi mezera v datech?)")
        return None
    times, preds, meta = predict_24h(tail.index, tail.to_numpy())
    return times, preds, meta, tail


# ---------------------------------------------------------------- denni beh

def denne(cur, out):
    from datetime import datetime, timezone

    icfg = load_config("influx", ("url", "org", "token", "bucket_history", "bucket_forecast"))
    cur.execute("SELECT measName FROM SmpMeasNameDB WHERE Id = ?", MEAS_ID)
    meter = cur.fetchone()[0]
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    to_ms = lambda dt: int((dt - epoch).total_seconds() * 1000)
    now = datetime.now(timezone.utc)
    out(f"\n=== Denni beh {now:%Y-%m-%d %H:%M} UTC: {meter} -> InfluxDB {icfg['url']} ===")

    # 1) odkud cist: posledni zapsana historie (- PREPSAT_DNI) a PREDIKCE_HISTORIE_DNI pred nejnovejsim zaznamem
    cur.execute("SELECT MAX(endTime) FROM UniArchiveBinPack WHERE keymeasName = ? AND keyArchID = ?", MEAS_ID, ARCH_ID)
    newest = cur.fetchone()[0].replace(tzinfo=timezone.utc)
    out(f"  nejnovejsi zaznam v DB: {newest:%Y-%m-%d %H:%M} UTC (stari dat {(now - newest).total_seconds() / 3600:.1f} h)")
    last_hist = influx_last_time(icfg, icfg["bucket_history"], "mereni", meter, "Avg.3P[kW]")
    if last_hist:
        hist_from = to_ms(last_hist) - PREPSAT_DNI * 86400000
        out(f"  posledni historie v Influxu: {last_hist:%Y-%m-%d %H:%M} UTC")
    else:
        hist_from = 0 if OD_HISTORIE is None else to_ms(datetime.fromisoformat(OD_HISTORIE).replace(tzinfo=timezone.utc))
        out(f"  bucket '{icfg['bucket_history']}' je prazdny - prvni beh, zapisuji historii "
            f"{'od zacatku dat' if OD_HISTORIE is None else 'od ' + OD_HISTORIE}")
    pred_from = to_ms(newest) - PREDIKCE_HISTORIE_DNI * 86400000
    read_from = min(hist_from, pred_from)

    # 2) MS SQL -> 1min zaznamy -> 15min okna
    windows, n_rec = aggregate_15min(read_records(cur, max(read_from, 0), out))
    out(f"  nacteno {n_rec} 1min zaznamu -> {len(windows)} uplnych 15min oken")
    if not windows:
        out("  Zadna data, konec.")
        return

    # 3) historie -> Influx
    lines = lines_history(windows, meter, hist_from)
    influx_write(icfg, icfg["bucket_history"], lines)
    out(f"  historie: zapsano {len(lines)} 15min oken do bucketu '{icfg['bucket_history']}'")

    # 4) predikce z poslednich PREDIKCE_HISTORIE_DNI dni
    res = forecast_from_windows(windows, pred_from, out)
    if res is None:
        return
    times, preds, meta, tail = res
    t0 = tail.index[-1]
    tag = f"predikce,meter={lp_escape(meter, '=')},predikce_od={t0:%Y-%m-%dT%H:%MZ}"
    lines = [f"{tag} {lp_escape('Avg.3P[kW]', '=')}={float(v)!r} {int(t.timestamp())}" for t, v in zip(times, preds)]
    influx_write(icfg, icfg["bucket_forecast"], lines)
    out(f"  predikce (model {meta['run_id']}): {len(lines)} kroku {times[0]:%Y-%m-%d %H:%M} .. "
        f"{times[-1]:%Y-%m-%d %H:%M} UTC, z historie {len(tail)} kroku, prumer {preds.mean():.1f} kW "
        f"-> bucket '{icfg['bucket_forecast']}'")


# ---------------------------------------------------------------- main

def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    print("Pripojuji se...")
    try:
        with connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT @@VERSION, SYSTEM_USER, CURRENT_USER, DB_NAME()")
            version, system_user, current_user, db = cur.fetchone()
            out(">>> Pripojeni FUNGUJE. <<<")
            out(version.splitlines()[0])
            out(f"Prihlaseny jako: SYSTEM_USER={system_user}  CURRENT_USER={current_user}  databaze={db}")

            if KROK == "denne":
                denne(cur, out)
            elif KROK == "export":
                export(cur, out)
            elif KROK == "prehled":
                explore(cur, out)
            elif KROK == "archiv":
                inspect_archive(cur, out)
            else:
                raise SystemExit(f"Neznamy KROK: {KROK!r}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n>>> CHYBA: {e} <<<")
        if DEBUG:
            print(f"Podrobny FreeTDS log: {LOG_PATH}")
        else:
            print("(podrobny log: nastav DEBUG = True v db.py a spust znovu)")
        sys.exit(1)

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nUlozeno do {OUT_PATH}")


if __name__ == "__main__":
    main()
