# Polymarket Weather Whale Finder

Identifies traders on Polymarket who appear to have early access to temperature
data ("fast-feed traders") by analyzing who gets into weather markets just before
large price moves.

## What it does

1. Fetches all active weather/temperature markets from Polymarket's Gamma API
2. Pulls full trade history for each market via the authenticated CLOB API
3. Detects hourly windows where prices moved significantly (≥5¢ by default)
4. Scores wallets by how often they traded in the correct direction in the 15
   minutes before each move
5. Prints a ranked report of likely fast-feed wallets

## Credentials needed

Three environment variables are required — these come from a Polymarket CLOB API
key, which is derived from a wallet signature (free, no funds needed):

```
POLY_API_KEY        e.g. abc123...
POLY_API_SECRET     e.g. xyz789...
POLY_PASSPHRASE     e.g. def456...
```

These map directly to the headers `POLY-API-KEY`, `POLY-API-SECRET`, and
`POLY-PASSPHRASE` on authenticated requests to `https://clob.polymarket.com`.

### How to generate them (one-time setup)

```bash
pip install py-clob-client
python - <<'EOF'
from py_clob_client.client import ClobClient
client = ClobClient(
    host="https://clob.polymarket.com",
    key="0xYOUR_PRIVATE_KEY",
    chain_id=137,
)
creds = client.create_or_derive_api_creds()
print("POLY_API_KEY =", creds.api_key)
print("POLY_API_SECRET =", creds.api_secret)
print("POLY_PASSPHRASE =", creds.api_passphrase)
EOF
```

## Setup

```bash
pip install -r requirements.txt
export POLY_API_KEY="..."
export POLY_API_SECRET="..."
export POLY_PASSPHRASE="..."
```

## Run

```bash
# Default: last 30 days, all active weather markets
python main.py

# Save raw trades and analysis to data/
python main.py --save

# Wider window, tighter signal threshold
python main.py --days 60 --min-move 0.03 --lead-minutes 10 --save
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--days` | 30 | Days of trade history to fetch |
| `--limit` | 50 | Max number of markets to pull |
| `--min-move` | 0.05 | Min price shift (0–1) to count as a move event |
| `--lead-minutes` | 15 | How many minutes before a move to look for early trades |
| `--save` | off | Write CSVs and JSON to `data/` |

## Output

Console report with two sections:

**Top wallets by volume** — who trades weather markets most actively.

**Fast-feed candidates** — wallets ranked by how many price-move events they
front-ran, total USD size of those early trades, and average lead time in minutes.
A wallet appearing here repeatedly across many events and markets is likely
receiving temperature data faster than the public feed.

## File structure

```
src/
  markets.py    — fetch weather events/markets from Gamma API
  trades.py     — fetch trades per token via authenticated CLOB API
  analysis.py   — price-move detection and early-mover scoring
main.py         — CLI entry point
data/
  raw/          — markets.json, trades_raw.json (with --save)
  processed/    — trades.csv, early_movers.csv (with --save)
```

## API endpoints used

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `gamma-api.polymarket.com/events` | None | Discover weather markets |
| `clob.polymarket.com/data/trades` | HMAC | Full trade history per token |
