## pull_kxhighlax_trades.R
## Pulls Kalshi trade tape for all bins of a single KXHIGHLAX event,
## aggregates into 1-minute bars, and computes trades-only OFI proxies
## per the CKS "TI" fallback described in PROJECT_CONTEXT.md.
##
## Outputs:
##   - bars_b72: 1-min bar table for the B72.5 bin (the winner)
##   - bars_all: same for all bins stacked
##   - Printed OFI table to console
##
## Dependencies: httr, jsonlite, dplyr, lubridate
## Run time: ~30s for a single event day (rate-limited 0.15s/page)

library(httr)
library(jsonlite)
library(dplyr)
library(lubridate)

## ── CONFIG ──────────────────────────────────────────────────────────────────

KALSHI_BASE  <- "https://external-api.kalshi.com/trade-api/v2"
EVENT_TICKER <- "KXHIGHLAX-26JUL02"        ## yesterday: change as needed
RATE_SLEEP   <- 0.15                        ## seconds between paginated calls
BAR_MINS     <- 1L                          ## bar width in minutes
## METAR at KLAX fires at :53 past each hour; sharp window is :52–:56
METAR_START_SEC <- 52L * 60L               ## seconds-past-hour start
METAR_END_SEC   <- 56L * 60L               ## seconds-past-hour end

## ── STEP 1: DISCOVER BINS for the event ─────────────────────────────────────
## Rather than guessing ticker suffixes, query the markets endpoint for the
## full event to get every bin that actually traded.

cat("Fetching market list for event:", EVENT_TICKER, "\n")

resp <- GET(
  paste0(KALSHI_BASE, "/markets"),
  query = list(event_ticker = EVENT_TICKER, limit = 100L)
)
stop_for_status(resp)
market_data <- fromJSON(content(resp, as = "text", encoding = "UTF-8"))
markets_df  <- as.data.frame(market_data$markets)

## Keep key fields; fall back gracefully if settled fields are absent
tickers <- markets_df$ticker
cat("Found", length(tickers), "bins:", paste(tickers, collapse = ", "), "\n\n")

## ── STEP 2: PULL TRADES for each bin ────────────────────────────────────────
## Uses GET /markets/trades?ticker=...&limit=1000 with cursor pagination.
## Returns newest-first; we reverse so rows are chronological.

pull_trades_for_ticker <- function(tkr) {
  all_trades <- list()
  cursor     <- ""
  page       <- 1L

  repeat {
    qry <- list(ticker = tkr, limit = 1000L)
    if (nchar(cursor) > 0) qry$cursor <- cursor

    r <- GET(paste0(KALSHI_BASE, "/markets/trades"), query = qry)
    stop_for_status(r)
    body <- fromJSON(content(r, as = "text", encoding = "UTF-8"))

    trades <- body$trades
    if (is.null(trades) || length(trades) == 0 || nrow(as.data.frame(trades)) == 0) break

    all_trades[[page]] <- as.data.frame(trades)
    cursor <- body$cursor %||% ""
    if (nchar(cursor) == 0) break

    page <- page + 1L
    Sys.sleep(RATE_SLEEP)
  }

  if (length(all_trades) == 0) return(NULL)

  ## bind pages, parse timestamps, sort chronologically
  df <- bind_rows(all_trades) |>
    mutate(
      created_time   = ymd_hms(created_time, tz = "UTC"),
      count_fp       = as.numeric(count_fp),
      yes_price      = as.numeric(yes_price_dollars),
      is_yes_taker   = (taker_outcome_side == "yes"),
      ticker         = tkr
    ) |>
    filter(!is.na(created_time)) |>
    arrange(created_time)

  df
}

## Null-coalescing operator (base R lacks %||%)
`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && a != "") a else b

cat("Pulling trades for each bin...\n")
all_bin_trades <- lapply(tickers, function(tkr) {
  cat(" ", tkr, "...")
  df <- pull_trades_for_ticker(tkr)
  if (is.null(df)) { cat(" (no trades)\n"); return(NULL) }
  cat(nrow(df), "trades\n")
  df
})
names(all_bin_trades) <- tickers
all_bin_trades <- Filter(Negate(is.null), all_bin_trades)

## ── STEP 3: BUILD 1-MIN BARS ─────────────────────────────────────────────────
## For each bar we compute:
##   bar_open / bar_close / bar_vwap  – price in [0,1]
##   yes_vol / no_vol / total_vol     – contract counts
##   taker_net   = yes_vol - no_vol   (positive = net YES aggression)
##   taker_share = yes_vol / total_vol
##   dP          = bar_close - bar_open (raw price change, cents = *100)
##   metar_flag  = TRUE if bar straddles :52–:56 past any hour
##
## beta_TI is estimated from a calm window (pre-signal, low-volume bars).
## We define "calm" as bars in the first third of the session with
## total_vol < median(total_vol). The TI regression is:
##   dP_cents ~ 0 + taker_net   (intercept forced to 0 per CKS)
## Residual z = (dP_cents - beta_TI * taker_net) / sd(calm_resids)

build_bars <- function(trades_df, bar_mins = BAR_MINS) {
  if (is.null(trades_df) || nrow(trades_df) == 0) return(NULL)

  trades_df |>
    mutate(
      ## floor to bar boundary
      bar_start = floor_date(created_time, paste(bar_mins, "minutes")),
      ## seconds-past-hour for METAR window flag
      secs_past_hour = as.integer(minute(created_time)) * 60L +
                       as.integer(second(created_time))
    ) |>
    group_by(ticker, bar_start) |>
    summarise(
      n_trades    = n(),
      bar_open    = first(yes_price),
      bar_close   = last(yes_price),
      bar_vwap    = weighted.mean(yes_price, count_fp),
      yes_vol     = sum(count_fp[is_yes_taker],  na.rm = TRUE),
      no_vol      = sum(count_fp[!is_yes_taker], na.rm = TRUE),
      total_vol   = sum(count_fp, na.rm = TRUE),
      ## METAR flag: TRUE if ANY trade in bar falls in the :52–:56 window
      metar_flag  = any(secs_past_hour >= METAR_START_SEC &
                          secs_past_hour <= METAR_END_SEC),
      .groups = "drop"
    ) |>
    mutate(
      taker_net   = yes_vol - no_vol,
      taker_share = yes_vol / pmax(total_vol, 1e-9),
      dP_cents    = (bar_close - bar_open) * 100
    ) |>
    arrange(bar_start)
}

## ── STEP 4: FIT beta_TI AND COMPUTE RESIDUAL z ──────────────────────────────
## Calm window: first 33% of bars AND total_vol < median(total_vol).
## beta_TI is the OLS slope of dP_cents ~ 0 + taker_net on calm bars.
## This is the trades-only analogue of the CKS OFI regression.

add_ofi_stats <- function(bars_df) {
  if (is.null(bars_df) || nrow(bars_df) < 5) return(bars_df)

  n          <- nrow(bars_df)
  calm_rows  <- seq_len(floor(n / 3))
  med_vol    <- median(bars_df$total_vol, na.rm = TRUE)

  calm_idx   <- intersect(
    calm_rows,
    which(bars_df$total_vol < med_vol & abs(bars_df$dP_cents) < 5)
  )

  if (length(calm_idx) < 3) {
    ## not enough calm bars — skip regression, fill NA
    bars_df$beta_TI    <- NA_real_
    bars_df$resid      <- NA_real_
    bars_df$resid_z    <- NA_real_
    bars_df$calm_resid_sd <- NA_real_
    return(bars_df)
  }

  calm_df   <- bars_df[calm_idx, ]
  fit       <- lm(dP_cents ~ 0 + taker_net, data = calm_df)
  beta_TI   <- coef(fit)[["taker_net"]]
  calm_sd   <- sd(residuals(fit))

  bars_df |>
    mutate(
      beta_TI       = beta_TI,
      calm_resid_sd = calm_sd,
      resid         = dP_cents - beta_TI * taker_net,
      resid_z       = resid / calm_sd
    )
}

## ── STEP 5: RUN PIPELINE ────────────────────────────────────────────────────

cat("\nBuilding 1-min bars and computing OFI stats...\n\n")

bars_list <- lapply(all_bin_trades, function(df) {
  tkr  <- df$ticker[1]
  bars <- build_bars(df)
  bars <- add_ofi_stats(bars)
  bars
})

bars_all <- bind_rows(bars_list)

## Focus printout on the winning bin (B72.5) during the trading day
## Filter to Jul 2 UTC only (pre-settlement cleanup on Jul 3 is noise)
bars_b72 <- bars_list[[grep("B72.5", names(bars_list))]] |>
  filter(as.Date(bar_start, tz = "UTC") == as.Date(EVENT_TICKER |>
    sub(".*-(\\d{2})(\\w{3})(\\d{2})$", "20\\1-\\3-\\2", x = _) |>
    as.Date(format = "%Y-%b-%d")))

## ── STEP 6: PRINT OFI TABLE ─────────────────────────────────────────────────
## Columns printed:
##   bar_start (UTC) | yes_vol | no_vol | taker_net | taker_share |
##   dP_cents | resid_z | metar_flag | bar_vwap

format_ofi_table <- function(bars, ticker_label) {
  cat("═══════════════════════════════════════════════════════════════════════\n")
  cat(" OFI TABLE:", ticker_label, " | bar width:", BAR_MINS, "min\n")
  cat(" beta_TI =", round(bars$beta_TI[1], 5),
      " calm_resid_sd =", round(bars$calm_resid_sd[1], 3), "\n")
  cat("═══════════════════════════════════════════════════════════════════════\n")

  display <- bars |>
    transmute(
      bar_UTC      = format(bar_start, "%H:%M"),
      yes_vol      = round(yes_vol, 0),
      no_vol       = round(no_vol, 0),
      taker_net    = round(taker_net, 0),
      tkr_share    = round(taker_share, 2),
      dP_c         = round(dP_cents, 1),
      resid_z      = round(resid_z, 2),
      vwap         = round(bar_vwap, 2),
      METAR        = ifelse(metar_flag, "<<METAR>>", "")
    )

  print(display, n = Inf)
  cat("\n")
}

format_ofi_table(bars_b72, paste0(EVENT_TICKER, "-B72.5 (settled YES)"))

## Save full table to CSV for further analysis
out_path <- file.path(
  dirname(rstudioapi::getSourceEditorContext()$path %||% "."),
  paste0("ofi_bars_", EVENT_TICKER, ".csv")
)
tryCatch(
  write.csv(bars_all, out_path, row.names = FALSE),
  error = function(e) message("Could not write CSV: ", conditionMessage(e))
)

cat("Done. bars_b72 and bars_all are in your environment.\n")
