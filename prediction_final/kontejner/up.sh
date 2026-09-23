#!/bin/sh
# Pomocny skript: precte aktualni nasazeny beh z runs/latest.json, predá ho
# jako RUN_ID do docker-compose (Dockerfile.api ho zabali do image) a
# postavi + spusti cely stack (influxdb + predikce-api + ingest).
#
# Pouziti:
#   ./up.sh          # postavi a spusti na popredi (Ctrl+C = stop)
#   ./up.sh -d        # na pozadi (detached)
#
# Vyzaduje prikaz `docker-compose` (na Debianu: `sudo apt install docker-compose`
# - viz README.md "Co jeste chybi").

set -e
cd "$(dirname "$0")"

RUN_ID=$(python3 -c "import json; print(json.load(open('../runs/latest.json'))['latest_run'])")
echo "Pouzivam beh RUN_ID=$RUN_ID (z ../runs/latest.json)"

export RUN_ID
docker-compose up --build "$@"
