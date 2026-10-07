# Produkční běh: MS SQL → predikce → InfluxDB (T12 P1.1)

Na Ubuntu VM běží každou hodinu (cron) `db.py`:

1. Přes Windows Authentication (NTLM, FreeTDS) čte z MS SQL databáze
   `VUT25` na `PC-EMVUT` binární archiv analyzátoru T12 P1.1.
2. Počítá 15min historii a 24h predikci `Avg.3P[kW]`. Model běží přímo
   v `db.py`, žádné API ani kontejnery.
3. Historii i predikci zapisuje do InfluxDB 2 na stejné VM.

Do MS SQL se nikdy nezapisuje, jen se z ní čte.

## Soubory

Na VM jsou ve stejné složce (teď `/home/tester/Plocha/predikce/v1/`):

| soubor | obsah |
|---|---|
| `db.py` | všechen kód (připojení, dekódování archivu, predikce, zápis do Influxu) |
| `dbtest.py` | test bez zápisu do Influxu: historie a predikce do CSV |
| `db_config.ini` | jen přihlašovací údaje (je v `.gitignore`, **ne**commitovat) |
| `model/` | natrénovaný model (`model.json`, `metadata.json`) = kopie `prediction_final/runs/<run_id>/` |

**Při aktualizaci kopíruj jen `db.py`, `dbtest.py` a případně `model/`.**
`db_config.ini` v projektu má prázdná hesla a přepsal by ten vyplněný na
VM. `db.py` a `dbtest.py` kopíruj vždy spolu, `dbtest.py` používá
funkce z `db.py`.

## Instalace (jednorázově)

### 1. Windows (PC-EMVUT)

SQL Server Configuration Manager:
1. *SQL Server Network Configuration → Protocols for SQLEXPRESS* →
   **TCP/IP → Enable**.
2. Dvojklik na TCP/IP → záložka *IP Addresses* → sekce **IPAll**:
   *TCP Dynamic Ports* = prázdné, *TCP Port* = `1433`.
3. *SQL Server Services* → **SQL Server (SQLEXPRESS) → Restart**.

(Pokud se VM nepřipojí, zkontrolovat ještě Windows Firewall – povolit
příchozí TCP 1433.)

### 2. Ubuntu VM – ovladač MS SQL

V tomto pořadí, `unixodbc` musí být před `pyodbc`:

```bash
sudo apt update
sudo apt install -y unixodbc unixodbc-dev freetds-bin tdsodbc python3-pip
sudo odbcinst -i -d -f /usr/share/tdsodbc/odbcinst.ini
odbcinst -q -d            # musí vypsat [FreeTDS]
pip3 install --user --break-system-packages pyodbc
```

Přidat server do `/etc/freetds/freetds.conf` (jeden příkaz):

```bash
printf '\n[ws11]\n\thost = 147.229.159.14\n\tport = 1433\n\ttds version = 7.4\n\tencryption = off\n' | sudo tee -a /etc/freetds/freetds.conf
```

### 3. Ubuntu VM – knihovny pro predikci

```bash
pip3 install --user --break-system-packages pandas numpy xgboost==2.1.3
```

### 4. Ubuntu VM – InfluxDB 2

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repos.influxdata.com/influxdata-archive.key | gpg --dearmor | sudo tee /etc/apt/keyrings/influxdata-archive.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/keyrings/influxdata-archive.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt update
sudo apt install -y influxdb2 influxdb2-cli
sudo systemctl enable --now influxdb
```

Ve webovém rozhraní `http://localhost:8086` (nebo přes `influx setup`)
založit organizaci, uživatele a buckety **`history`** a **`forecast`**
a vytvořit API token se čtením i zápisem do bucketů.

### 5. `db_config.ini`

Když soubor chybí, první spuštění `db.py` ho vytvoří jako šablonu. Když v
něm chybí jen sekce `[influx]`, `db.py` ji doplní sám a skončí s výzvou
k vyplnění.

```ini
[mssql]
servername = ws11
database = VUT25
driver = FreeTDS
uid = PC-EMVUT\EM VUT
pwd = heslo_k_windows_uctu

[influx]
url = http://localhost:8086
org = nazev_organizace
token = token_z_influxu
bucket_history = history
bucket_forecast = forecast
```

Hodnoty se píšou bez uvozovek. Zpětné lomítko, mezera i `;` jsou v
pořádku.

## Spuštění

```bash
python3 db.py                  # provede KROK nastaveny v db.py (výchozí "denne")
KROK=export python3 db.py      # jednorázově jiný krok, bez úpravy souboru
python3 dbtest.py              # test: historie + predikce do CSV, do Influxu nic
```

| `KROK` | co dělá |
|---|---|
| `"denne"` (výchozí) | nová data z MS SQL → 15min historie + 24h predikce → InfluxDB (spouští cron každou hodinu) |
| `"export"` | celá historie místa `MEAS_ID` do `T12_P1.1_15min.csv` (formát jako `data_processed/main_archive_15min.csv`) |
| `"prehled"` | tabulky, sloupce a ukázkové řádky celé databáze |
| `"archiv"` | rozbor binárního archivu jednoho místa (definice, balíky, hex) |

Výpis posledního běhu se ukládá do `db_out.txt` (u `dbtest.py` do
`dbtest_out.txt`).

Při chybě připojení nastav v `db.py` proměnnou `DEBUG = True`. Skript
pak zapíše podrobný FreeTDS log do `db_freetds.log`, podle kterého se
dá chyba dohledat v sekci Historie problémů níže.

### Pravidelný běh (`KROK = "denne"`, cron každou hodinu)

Krok se jmenuje `denne` z historických důvodů, spouští se každou hodinu.
Jeden běh:

1. Zjistí v Influxu čas poslední uložené historie.
2. Z MS SQL načte nové 1min záznamy (od poslední historie minus
   `PREPSAT_DNI` = 1 den, a vždy aspoň 8 dní pro model). Spočítá z nich
   15min okna v UTC. Poslední neúplné okno se zapíše až při dalším běhu.
3. **Historie** → bucket `history`, measurement `mereni`, tag
   `meter=T12 P1.1`, 30 polí se stejnými názvy jako sloupce CSV
   (`Avg.3P[kW]`, `Avg.U1[V]`, …).
4. **Predikce** → bucket `forecast`, measurement `predikce`, pole
   `Avg.3P[kW]`, 96 kroků po 15 min navazujících na poslední úplné okno
   v DB. Tag `predikce_od` je čas posledního skutečného okna, takže se
   predikce z různých dnů navzájem nepřepisují.

**Časy:** každý 1min záznam má v DB vlastní časovou značku. Okno se
uloží s časem svého začátku v UTC (okno 14:15 = minuty 14:15–14:29).
Mezery v datech se nijak nedoplňují, okna za tu dobu prostě chybí.

**První běh (prázdný Influx):** načte z MS SQL celou historii (od
21.09.2024, asi 1 milion 1min záznamů, přes 1 GB dat, trvá několik
minut) a zapíše asi 65 tisíc 15min oken. Pokud stačí kratší historie,
nastav v `db.py` proměnnou `OD_HISTORIE` (např. `"2025-01-01"`, od kdy
se trénoval model). Predikce vznikne jen jedna, zpětné predikce za
minulé dny se nepočítají.

**Další běhy (i po výpadku, např. po 8 dnech):** doplní historii za
celou dobu od posledního běhu a vytvoří jednu novou 24h predikci.

**Když od minulého běhu nepřibyla nová data** (DB se neaktualizuje
každou hodinu), vyjde stejný `predikce_od` a predikce se jen přepíše
stejnými hodnotami. V Influxu tedy přibude nová predikce jen tehdy, když
v DB přibudou nová data. Historie se taky jen přepíše stejnými
hodnotami.

### Model

- Výpočet je stejný jako v `prediction_final/api` (blend 0,8 × XGBoost
  + 0,2 × hodnota před 24 h, každý jako vlastní 24h rekurze), ověřeno
  na shodu do 0,001 kW.
- **Rozdíl proti API:** kalendářní příznaky (hodina, den v týdnu,
  svátky) se počítají v místním čase (`CAS_FEATUR`), stejně jako při
  tréninku. Backtest na srpnu 2026 dává MAE 24h 19,37 kW, v UTC by to
  bylo 22,23 kW.
- Model se načítá při každém běhu. Nový model = přepsat `model.json` a
  `metadata.json` ve `model/`. Který model běží, je vidět ve výpisu
  (`predikce (model 20260923_095645)`).
- Na víkendy a svátky model předpovídá plochý průběh bez denních špiček
  (např. neděle 4.10.2026: 123–141 kW). To je očekávané chování.

### Cron

`crontab -e`, běh každou hodinu (v celou):

```
0 * * * * cd /home/tester/Plocha/predikce/v1 && /usr/bin/flock -n /tmp/kmb_predikce.lock /usr/bin/python3 db.py >> denne.log 2>&1
```

`flock -n` zajistí, že se nový běh nespustí, dokud předchozí ještě
běží. To může nastat hlavně u prvního běhu, který načítá celou historii.
Běžný hodinový běh čte jen asi 9 dní dat, takže je výrazně kratší.

Kontrola běhu: `tail -50 denne.log`, výsledek posledního běhu je i v
`db_out.txt`.

## Jak jsou data v DB uložená

- Tabulka `UniArchiveBinPack`, místo `keymeasName = 3` (T12 P1.1) a
  `keyArchID = 0` (hlavní archiv, 1 min). Balík odpovídá zhruba měsíci
  až čtvrtletí. Data jsou ve sloupci `Data`, kde jsou záznamy pevné
  délky za sebou.
- Záznam začíná 8 B časem (ms od 2000-01-01 UTC, big-endian). Za ním
  jsou pole typu `UniPropDesc` v pořadí podle XML v
  `UniArchiveDefinition.TypeXML` (podle `ArchDef` balíku). Pole
  `VirtualPropDesc` se dopočítávají a v datech nejsou.
- Velikost pole určuje `TypeSer`, ne `<len>`: například `Reset` (UInt16)
  má `len=4`, ale v datech zabírá 2 B. U polí (`Single[]`) je `<len>`
  počet prvků.
- Výkony jsou ve W, `db.py` je převádí na kW, kvar a kVA.
- Formát záznamu se během času měnil (ArchDef 4 → 18 → 7, s 781 → 2006
  → 2342 B), proto se rozložení čte z XML pro každý balík zvlášť.

## Podrobnosti k připojení (proč jsou kroky takové)

- **Připojuje se přes jméno sekce (`SERVERNAME=ws11`), ne přes
  `SERVER=ip`.** Když je v connection stringu `SERVER=`, FreeTDS soubor
  `freetds.conf` vůbec nečte, takže by se neuplatnilo `encryption = off`.
- **`encryption = off`** znamená, že se nedělá TLS handshake (v logu
  `detected crypt flag 2`). Tím odpadá problém č. 4 níže. Heslo po síti
  nejde (NTLM posílá jen challenge-response), data dotazů ale jdou
  nešifrovaně. Pro interní síť to stačí.
- **Formát `UID`:** pro lokální účet je před jménem přesný hostname
  stroje se SQL Serverem (`PC-EMVUT\...`). Mezera ve jméně nevadí,
  hodnota se v connection stringu zabalí do `{}`.
- **Kdyby někdy bylo potřeba šifrování zapnout** (`encryption = require`),
  je nutné na VM omezit GnuTLS na TLS 1.2:
  ```bash
  sudo mkdir -p /etc/gnutls
  printf '[overrides]\ndefault-priority-string = NORMAL:-VERS-TLS1.3\n' | sudo tee /etc/gnutls/config
  ```
- `logic error: cannot change query state from IDLE to PENDING` ve
  FreeTDS logu je neškodná hláška po zrušení zbytku výsledku.

## Historie problémů

Tyto problémy se objevily postupně a návod výše je už řeší:

1. **`ImportError: libodbc.so.2`**: `pyodbc` nainstalovaný přes pip bez
   systémové knihovny `unixodbc`. Řešení: instalovat `unixodbc` před
   `pyodbc` (Instalace, krok 2).
2. **Named instance (`PC-EMVUT\SQLEXPRESS`) + SQL Browser (UDP 1434)
   timeoutoval** (`tds7_get_instance_port: timed out`). FreeTDS se musí
   nejdřív zeptat SQL Browseru na port instance a ten dotaz z VM nikdy
   nedostal odpověď. Řešení: pevný port 1433 a připojení přímo na něj,
   bez jména instance (Instalace, kroky 1 a 2).
3. **TCP/IP protokol byl na SQLEXPRESS vypnutý** a nastavený na
   „dynamic port“ (náhodný při každém restartu). Řešení: Instalace,
   krok 1.
4. **`handshake failed: Error in the pull function` / `Unexpected EOF
   from the server`**: novější GnuTLS na VM nabízí TLS 1.3, kterému
   Windows TLS vrstva pod SQL Serverem nerozumí a spojení zahodí.
   Řešení: `encryption = off` ve `freetds.conf` (viz Podrobnosti).

## Stav a ověření

- **5.10.2026 připojení funguje:** SQL Server 2022 Express
  (16.0.1000.6), NTLM login jako `PC-EMVUT\EM VUT`
  (`CURRENT_USER=dbo`), databáze `VUT25`.
- **5.10.2026 export ověřen** proti `data_processed/main_archive_15min.csv`
  v celém překryvu (21.09.2024–01.09.2026, 61 929 společných 15min
  oken). Medián rozdílu Avg.3P je 0,00005 kW (jen zaokrouhlení CSV).
  Shoda platí přes všechny tři formáty záznamu i přechody na letní čas.
  Liší se jen 6 oken: přechod na zimní čas 26.10.2025 02:00 (0,8 kW),
  krajní okna dvou mezer v CSV v květnu 2026 (DB má o 30 oken víc) a
  poslední neúplné okno CSV.
- **5.10.2026 `dbtest.py` na VM:** historie (1 344 oken, 20.9.–4.10.)
  sedí s exportem, predikce sedí s přepočtem na desktopu (rozdíl do
  0,75 kW kvůli jiné verzi XGBoostu).

## Co zbývá vyřešit

- **Zpoždění dat v DB:** záznamy do MS SQL nepřicházejí průběžně. 5.10.
  byl nejnovější záznam ze 4.10. 03:22 místního času, tedy zhruba 40 h
  starý. Predikce navazuje na poslední data v DB, takže při takovém
  zpoždění pokrývá už uplynulý čas. Je potřeba zjistit, jak často
  software analyzátoru data do DB stahuje.
- Účet má `dbo` (plná práva). Pro produkční běh zvážit vlastní login jen
  s `db_datareader` na `VUT25`.
- `prediction_final/api` počítá kalendářní příznaky v UTC, ale model
  byl trénovaný na místním čase (MAE 22,23 kW místo 19,37 kW). Týká se
  jen API, denní běh v `db.py` to má správně.
