# Připojení k MS SQL databázi (Windows Authentication, z Ubuntu VM)

Cíl: z Ubuntu VM (běžící na stejném stroji jako SQL Server, `PC-EMVUT`) se
přes Windows Authentication (NTLM) připojit k databázi `VUT25`
(SQLEXPRESS instance), bez přidávání VM do AD domény a bez Kerberos.
Používá se obyčejný uživatel/heslo — stejný typ účtu jako přihlášení do
Windows (lokální účet stroje nebo doménový), jen FreeTDS ho posílá přes
NTLM protokol místo SQL-login autentizace.

## Rychlý postup (ověřeno 2026-10-05)

### 1. Windows (PC-EMVUT), jednorázově

SQL Server Configuration Manager:
1. *SQL Server Network Configuration → Protocols for SQLEXPRESS* →
   **TCP/IP → Enable**.
2. Dvojklik na TCP/IP → záložka *IP Addresses* → sekce **IPAll**:
   *TCP Dynamic Ports* = prázdné, *TCP Port* = `1433`.
3. *SQL Server Services* → **SQL Server (SQLEXPRESS) → Restart**.

(Pokud se VM nepřipojí, zkontrolovat ještě Windows Firewall – povolit
příchozí TCP 1433.)

### 2. Ubuntu VM, jednorázově

Instalace (v tomto pořadí, `unixodbc` musí být před `pyodbc`):

```bash
sudo apt update
sudo apt install -y unixodbc unixodbc-dev freetds-bin tdsodbc python3-pip
sudo odbcinst -i -d -f /usr/share/tdsodbc/odbcinst.ini
odbcinst -q -d            # musí vypsat [FreeTDS]
pip3 install pyodbc       # na novějším Ubuntu případně: pip3 install --break-system-packages pyodbc
```

Přidat server do `/etc/freetds/freetds.conf` (jeden příkaz):

```bash
printf '\n[ws11]\n\thost = 147.229.159.14\n\tport = 1433\n\ttds version = 7.4\n\tencryption = off\n' | sudo tee -a /etc/freetds/freetds.conf
```

### 3. Soubory na VM

Na VM jsou potřeba ve stejné složce:

| soubor | obsah |
|---|---|
| `db.py` | všechen kód |
| `db_config.ini` | jen přihlašovací údaje (je v `.gitignore`, **ne**commitovat) |
| `model/` | natrénovaný model (`model.json`, `metadata.json`) = kopie `prediction_final/runs/<run_id>/` |

`db_config.ini` stačí vyplnit. Když chybí, první spuštění `db.py` ho
vytvoří jako šablonu. Když v něm chybí jen sekce `[influx]`, `db.py` ji
doplní sám a skončí s výzvou k vyplnění:

```ini
[mssql]
servername = ws11
database = VUT25
driver = FreeTDS
uid = PC-EMVUT\EM VUT
pwd = tvoje_heslo

[influx]
url = http://localhost:8086
org = nazev_organizace
token = token_z_influxu
bucket_history = history
bucket_forecast = forecast
```

Hodnoty se píšou bez uvozovek. Zpětné lomítko, mezera i `;` jsou v
pořádku.

Knihovny pro predikci (jednorázově):

```bash
pip3 install --user --break-system-packages pandas numpy xgboost==2.1.3
```

### 4. Spuštění

```bash
python3 db.py                  # provede KROK nastaveny v db.py (vychozi "denne")
KROK=export python3 db.py      # jednorazove jiny krok, bez upravy souboru
```

| `KROK` | co dělá |
|---|---|
| `"denne"` (výchozí) | nová data z MS SQL → 15min historie + 24h predikce → InfluxDB (spouští cron) |
| `"export"` | celá historie místa `MEAS_ID` do `T12_P1.1_15min.csv` (formát jako `data_processed/main_archive_15min.csv`) |
| `"prehled"` | tabulky, sloupce a ukázkové řádky celé databáze |
| `"archiv"` | rozbor binárního archivu jednoho místa (definice, balíky, hex) |

Z MS SQL se jen čte. Výpis posledního běhu se ukládá do `db_out.txt`.

Při chybě připojení nastav v `db.py` proměnnou `DEBUG = True`. Skript
pak zapíše podrobný FreeTDS log do `db_freetds.log`, podle kterého se
dá chyba dohledat v sekci Historie problémů níže.

### 5. Denní běh (cron)

Jeden běh `KROK = "denne"`:

1. Zjistí v Influxu čas poslední uložené historie.
2. Z MS SQL načte nové 1min záznamy (plus poslední den znovu a 8 dní
   pro model) a spočítá z nich 15min okna v UTC. Poslední neúplné okno
   se zapíše až při dalším běhu.
3. **Historie** → bucket `history`, measurement `mereni`, tag
   `meter=T12 P1.1`, pole stejná jako sloupce CSV (`Avg.3P[kW]`, …).
   Při prvním běhu (prázdný bucket) se zapíše celá historie z DB, nebo
   od data v proměnné `OD_HISTORIE`.
4. **Predikce** → bucket `forecast`, measurement `predikce`, pole
   `Avg.3P[kW]`, 96 kroků po 15 min navazujících na poslední úplné okno.
   Tag `predikce_od` je čas posledního skutečného okna, takže se
   predikce z různých dnů navzájem nepřepisují.

**První běh (prázdný Influx):** načte z MS SQL celou historii (od
21.09.2024, asi 1 milion 1min záznamů, přes 1 GB dat, trvá několik
minut) a zapíše asi 65 tisíc 15min oken. Pokud stačí kratší historie,
nastav v `db.py` proměnnou `OD_HISTORIE` (např. `"2025-01-01"`, od kdy
se trénoval model). Predikce vznikne při prvním běhu jen jedna: 24 h
navazujících na poslední úplné 15min okno v DB. Zpětné predikce za
minulé dny se nepočítají.

**Další běhy:** znovu zapíšou poslední den historie (`PREPSAT_DNI`),
přidají nová okna a vytvoří jednu novou 24h predikci.

Model se počítá přímo v `db.py`, API se nevolá. Výpočet je ověřený
proti `prediction_final/api` (shoda na 0,001 kW). Jediný rozdíl je, že
kalendářní příznaky se počítají v místním čase (`CAS_FEATUR`), stejně
jako při tréninku. Backtest na srpnu 2026 dává MAE 24h 19,37 kW (v UTC
by to bylo 22,23 kW).

Nastavení cronu (`crontab -e`), běh každý den v 6:00:

```
0 6 * * * cd /home/tester/Plocha/predikce/test_pripojeni && /usr/bin/python3 db.py >> denne.log 2>&1
```

Cestu uprav podle toho, kde `db.py` na VM leží. Kontrola běhu:
`tail -50 denne.log`, výsledek posledního běhu je i v `db_out.txt`.

### Jak jsou data v DB uložená

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
- Ověřeno proti CSV exportu: záznam z 21.09.2024 14:29 UTC dává stejné
  hodnoty jako CSV pro 16:29 místního času (Avg.3P 128,043 kW,
  THDI1 19,094 % atd.).

## Podrobnosti (proč jsou kroky takové)

- **Připojuje se přes jméno sekce (`SERVERNAME=ws11`), ne přes
  `SERVER=ip`.** Když je v connection stringu `SERVER=`, FreeTDS soubor
  `freetds.conf` vůbec nečte, takže by se neuplatnilo `encryption = off`.
- **`encryption = off`** znamená, že se nedělá TLS handshake (v logu
  `detected crypt flag 2`). Tím odpadá problém č. 4 (GnuTLS × Windows
  TLS). Heslo po síti nejde (NTLM posílá jen challenge-response), data
  dotazů ale jdou nešifrovaně.
- **Pořadí balíčků:** `unixodbc` musí být nainstalovaný před `pyodbc`,
  jinak `import pyodbc` spadne na `ImportError: libodbc.so.2`.
- **Formát `UID`:** pro lokální účet je před jménem přesný hostname
  stroje se SQL Serverem (`PC-EMVUT\...`). Mezera ve jméně nevadí,
  hodnota se v connection stringu zabalí do `{}`.
- **Kdyby někdy bylo potřeba šifrování zapnout** (`encryption = require`),
  je nutné na VM omezit GnuTLS na TLS 1.2 (viz problém č. 4):
  ```bash
  sudo mkdir -p /etc/gnutls
  printf '[overrides]\ndefault-priority-string = NORMAL:-VERS-TLS1.3\n' | sudo tee /etc/gnutls/config
  ```

## Historie problémů (proč je postup takový, jaký je)

Pro budoucí referenci — tyto problémy se objevily postupně, v tomto
pořadí, a návod výše už je zahrnuje:

1. **`ImportError: libodbc.so.2`** — `pyodbc` nainstalovaný přes pip bez
   systémové knihovny `unixodbc`. → fix: instalovat `unixodbc` před
   `pyodbc` (krok B1).
2. **Named instance (`PC-EMVUT\SQLEXPRESS`) + SQL Browser (UDP 1434)
   timeoutoval** (`tds7_get_instance_port: timed out`) — FreeTDS se
   nejdřív musí dotázat SQL Browser služby na skutečný TCP port instance,
   a ten dotaz z VM nikdy nedostal odpověď (firewall/síť). → fix: zjistit
   přímo z SQL Serveru, jaký port používá, a připojit se natvrdo na
   konkrétní `SERVER:PORT`, bez jména instance.
3. **TCP/IP protokol byl na SQLEXPRESS úplně vypnutý** (zjištěno v SQL
   Server Configuration Manager) — proto žádný TCP port ani neexistoval,
   dokud se nezapnul. Navíc byl nastavený na "dynamic port" (náhodný při
   každém restartu), což dělalo problém nepředvídatelným. → fix: krok A
   (enable TCP/IP, nastavit pevný port 1433, restart služby).
4. **`handshake failed: Error in the pull function` / `Unexpected EOF
   from the server`** — i po zprovoznění TCP spojení na portu 1433 spadl
   TLS handshake. Příčina: FreeTDS na Ubuntu VM používá novější GnuTLS,
   která v "Client Hello" nabízí TLS 1.3 a velmi nové kryptografické
   algoritmy (vč. post-kvantových podpisů) — starší Windows TLS/Schannel
   vrstva pod SQL Serverem tomu nerozumí a spojení rovnou zahodí, místo
   aby odpověděla/vyjednala nižší verzi. → fix: krok B2 (omezit GnuTLS na
   max. TLS 1.2 systémovým override souborem).

## Stav

**2026-10-05: spojení FUNGUJE.** test připojení z Ubuntu VM:
SQL Server 2022 Express (16.0.1000.6, RTM), NTLM login jako
`PC-EMVUT\EM VUT` (`CURRENT_USER=dbo`), databáze `VUT25`.

Poznámky z FreeTDS logu:
- `detected crypt flag 2` = server šifrování nepodporuje/nevyžaduje →
  provoz po síti jde **nešifrovaně** (heslo ne — NTLM posílá jen
  challenge-response, ale data dotazů ano). Pro interní síť OK.
- `logic error: cannot change query state from IDLE to PENDING` je
  neškodná hláška FreeTDS po zrušení zbytku výsledku (pyodbc cancel).

**2026-10-05: export ověřen** proti `data_processed/main_archive_15min.csv`
v celém překryvu (21.09.2024–01.09.2026, 61 929 společných 15min oken).
Medián rozdílu Avg.3P je 0,00005 kW, 99. percentil 0,0002 kW (jen
zaokrouhlení CSV na 3 desetinná místa). Shoda platí i přes všechny tři
formáty záznamu a přechody na letní čas. Odlišných je jen 6 oken:
- 26.10.2025 02:00 (přechod na zimní čas, duplicitní hodina): 0,8 kW,
- 08.05.2026 16:15–18:30 a 23.05.2026 09:00–13:45: v CSV mezera, DB
  data má (30 oken navíc). Krajní okna mezer se liší, protože v CSV
  jsou neúplná,
- 01.09.2026 00:00: v CSV poslední, neúplné okno.

DB navíc obsahuje data za 01.09.–04.10.2026, která v CSV nejsou.

## Co zbývá vyřešit

- Účet má `dbo` (plná práva). Pro produkční skript zvážit vlastní
  login jen s `db_datareader` na `VUT25`.
- **Zpoždění dat v DB:** záznamy do MS SQL nepřicházejí průběžně. Při
  exportu 5.10. byl nejnovější záznam z 4.10. 01:22 UTC, tedy zhruba
  39 h starý. Predikce navazuje na poslední data v DB, takže při takovém
  zpoždění pokrývá už uplynulý čas. Je potřeba zjistit, jak často
  software analyzátoru data do DB stahuje.
- `prediction_final/api` počítá kalendářní příznaky v UTC, ale model
  byl trénovaný na místním čase (MAE 22,23 kW místo 19,37 kW). Týká se
  jen API, denní běh v `db.py` to má správně.
