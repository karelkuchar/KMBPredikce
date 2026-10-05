# Příklady požadavků/odpovědí

Skutečné (nevymyšlené) páry požadavek/odpověď, vygenerované reálným
voláním běžícího `api.py` na reálných datech projektu.

| soubor | popis |
|---|---|
| `request_short.json` / `response_short.json` | minimální platný požadavek — přesně 192 záznamů (48h, dolní hranice) |
| `request_long.json` / `response_long.json` | delší požadavek — 604 záznamů (~6.3 dne), blíže doporučeným 7 dnům |
| `request_with_history.json` / `response_with_history.json` | stejná historie jako `request_short.json` (192 záznamů), navíc `"include_history": true` — odpověď proto obsahuje i pole `history` (použitá historie) |

Všechny odpovědi mají 96 kroků predikce (24h dopředu, 15min krok).
