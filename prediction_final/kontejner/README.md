# Kontejnerizace predikčního API + InfluxDB

Tři samostatné kontejnery (`docker-compose.yml`):

| Kontejner | Co dělá | Co je uvnitř |
|---|---|---|
| `predikce-api` | FastAPI `/predict` — dostane historii, vrátí 24h predikci. Bezstavové, nic sám neukládá. | **model je zabalený přímo v image** (`COPY runs/<run_id>/`), žádná závislost na Influxu ani na zbytku repa — je to ten kontejner, co se bude přesouvat mimo, proto je plně samostatný |
| `influxdb` | úložiště časových řad | oficiální `influxdb:2` image, data v pojmenovaném volume `influxdb_data` |
| `ingest` | **jediný, kdo píše do Influxu** | `ingest.py` — FastAPI služba na portu 8001, čeká, až ji **někdo zvenku zavolá** s novými daty (`POST /ingest`) |

**Tenhle kontejner nemá přístup k žádným datům sám** — nemontuje žádné CSV,
nic sám nestahuje. Kdo měření má (analyzátor / jiný systém), ten je na
`ingest` **pošle** (push, ne pull). Při každém přijatém volání `ingest`:

1. uloží poslaná měření do Influxu (`skutecnost`),
2. dotáže se Influxu na posledních 48h `skutecnost` pro daný měřič (Influx
   tak funguje jako sdílená paměť — nezáleží, jestli volající pošle 1 nový
   bod, nebo rovnou celých 48h najednou),
3. pokud je historie dost dlouhá (≥192 záznamů), zavolá `predikce-api` a
   vrácených 96 bodů uloží do Influxu (`predikce`, otagováno `predicted_at`
   = kdy predikce vznikla, aby se při dalším volání nepřepisovaly staré
   predikce na stejných budoucích časech).

```
volající (má data) --POST /ingest--> [ingest] --zapíše "skutečnost"--> [influxdb]
                                         |
                                         +--POST /predict--> [predikce-api]
                                         |                    (bezstavové, model uvnitř)
                                         +--zapíše "predikce"--> [influxdb]
```

## Základní pojmy (pokud jste Docker ještě nepoužívali)

- **Image** = "zabalený" balíček (kód + Python + knihovny), postavený z
  `Dockerfile`. Image sama o sobě neběží, je to jen šablona.
- **Kontejner** = běžící instance image. Z jedné image jich lze spustit
  víc najednou.
- **`docker build`** = postaví image z Dockerfile.
- **`docker run`** = spustí jeden kontejner z image.
- **`docker-compose`** = spustí/postaví víc kontejnerů najednou podle
  `docker-compose.yml` (přesně náš případ — 3 služby).
- **Volume** = trvalé úložiště mimo kontejner. Kontejner sám je "jednorázový"
  (po smazání zmizí i to, co si zapsal dovnitř) — Influx databáze proto musí
  být ve volume (`influxdb_data`), jinak by při každém restartu zmizela.

## Instalace (předpoklady)

Potřeba jsou dva nástroje: `docker` a `docker-compose`.

```bash
docker --version           # overeni, ze uz je docker nainstalovany
docker-compose --version   # a docker-compose take
```

Pokud `docker` chybí:
```bash
sudo apt install docker.io
```

Pokud chybí `docker-compose` (na Debianu je moderní `docker compose` jako
podpříkaz jen v Dockerově vlastním APT repozitáři, který by bylo nutné
zvlášť přidávat — jednodušší je starší samostatný `docker-compose` přímo
z Debian repozitářů):
```bash
sudo apt install docker-compose
```

Aby navíc šel `docker` používat bez `sudo` u každého příkazu:
```bash
sudo usermod -aG docker $USER
```
(pak je potřeba se odhlásit/znovu přihlásit, ať se to projeví — do té doby
je nutné psát `sudo docker-compose ...` / `sudo ./up.sh`).

## Spuštění

```bash
cd prediction_final/kontejner
./up.sh              # postaví images + spustí vše na popředí (Ctrl+C = zastaví)
./up.sh -d            # totéž, ale na pozadí
```

`up.sh` si sám přečte, jaký model je aktuálně nasazený
(`prediction_final/runs/latest.json`), a předá ho jako `RUN_ID` do
`docker build`, aby se zabalil do `predikce-api` image.

**Užitečné příkazy, jakmile to běží:**

```bash
docker-compose ps                    # co běží
docker-compose logs -f ingest        # živé logy jednoho kontejneru (Ctrl+C = jen odpojí sledování)
docker-compose logs -f predikce-api
docker-compose exec predikce-api sh  # shell dovnitř běžícího kontejneru (ladění)
docker-compose down                  # zastaví a smaže kontejnery (volume s daty Influxu zůstane)
docker-compose down -v               # totéž + smaže i volume (ztratíš historii v Influxu!)
```

Přímý přístup k API a Influxu:
- `predikce-api`: `http://localhost:8000` (`/health`, `/predict`, `/docs` —
  FastAPI má vestavěné interaktivní swagger UI)
- `influxdb` UI: `http://localhost:8086` (uživatel/heslo/token viz
  `docker-compose.yml` — **pro produkci je nutné je změnit**, teď jsou tam
  jen vývojové placeholdery)

## Po přetrénování modelu

Model je zabalený v image, takže pouhý restart kontejneru nestačí —
je potřeba **znovu postavit `predikce-api` image**:

```bash
python3 prediction_final/train.py     # natrénuje nový běh, aktualizuje runs/latest.json
cd prediction_final/kontejner
./up.sh -d                             # znovu přečte runs/latest.json a přestaví image s novým RUN_ID
```

`ingest` a `influxdb` přestavovat netřeba, ty na modelu nezávisí.

## Vyzkoušení nanečisto (a úklid)

Postup pro první vyzkoušení, včetně toho, jak po sobě všechno zase smazat.
Předpokládá nainstalované `docker` + `docker-compose` (viz "Instalace" výš).

**1. Spusťte stack:**
```bash
cd prediction_final/kontejner
./up.sh -d
docker-compose ps          # overeni, ze bezi vsechny 3 kontejnery (predikce-api, influxdb, ingest)
```

**2. Otestujte `predikce-api` samostatně** (ověří, že model uvnitř image funguje):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @../api/example_json/request_short.json | head -c 300
```
Použitý JSON je existující reálný testovací příklad
(`api/example_json/request_short.json`, přesně 192 záznamů = 48h) — stejný
formát `{"readings": [...]}` platí i pro `ingest`.

**3. Otestujte celý tok přes `ingest`** (uloží do Influxu skutečnost i predikci):
```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d @../api/example_json/request_short.json
```
Odpověď by měla mít `"prediction_made": true` (rovnou 192 záznamů = dost
historie). Volání s méně záznamy vrátí `"prediction_made": false` a hlášku
o nedostatku historie.

**4. Volitelně nahlédněte do Influxu:** `http://localhost:8086`, přihlásit
se `admin` / `changeme123` (z `docker-compose.yml`), Data Explorer → bucket
`predikce` → measurementy `skutecnost` a `predikce`.

**5. Sledování logů, pokud něco nesedí:**
```bash
docker-compose logs -f ingest
docker-compose logs -f predikce-api
```

**6. Úklid po dokončení testu:**
```bash
docker-compose down -v          # zastaví + smaže kontejnery A volume (i data v Influxu)
docker image rm predikce-api predikce-ingest influxdb:2   # smaže i postavené/stažené images
```
`docker-compose down` bez `-v` by nechal volume `influxdb_data` ležet — pro
"jen na zkoušku, pak smazat" je potřeba `-v`, ať nezůstane nic po sobě.

## Co ještě chybí / je potřeba doladit (otevřené otázky)

- **Kdo a jak volá `POST /ingest`?** Zatím to v projektu nikde není
  automatizované — potřeba domluvit s tím, kdo má přístup k měřidlu/
  analyzátoru, jak často a v jakém formátu bude data posílat (jeden bod po
  15 min? dávka za více hodin najednou po výpadku?). Formát requestu je
  stejný jako u `predikce-api` (`readings: [{time, value_kW}]`, čas musí
  být UTC).
- **Přístupová práva k Dockeru:** pokud uživatel není v `docker` skupině,
  `docker`/`docker-compose` vyžadují `sudo` u každého příkazu — viz
  "Instalace" výš, sekce s `usermod -aG docker`.
- **Produkční tajemství:** Influx token/heslo v `docker-compose.yml` jsou
  vývojové placeholdery — pro ostrý provoz patří do `.env` souboru (mimo
  git) nebo secret manageru, ne natvrdo v compose souboru.
- **Zabezpečení `/ingest`:** zatím bez autentizace — kdokoliv s přístupem
  na port 8001 může zapisovat "skutečná" data. Pokud bude port vystavený
  mimo důvěryhodnou síť, přidat autentizaci (API klíč apod.).
