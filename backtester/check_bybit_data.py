#!/usr/bin/env python3
"""Smoke test the Bybit-backed backtester data pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from marketdata import DataRequirement, MarketDataRequest

from .data import BybitClient
from .models import MarketType
from .pipeline import prepare_market_context


def main() -> None:
    client = BybitClient(market_type=MarketType.FUTURES)
    symbol = "BTC/USDT"
    end = datetime.now(UTC)
    start = end - timedelta(days=2)

    klines = client.fetch_klines(symbol, "1h", start, end)
    print(f"OHLCV ok: {len(klines)} candle(s)")
    if klines:
        print(f"Last close: {klines[-1].close} @ {klines[-1].close_time.isoformat()}")

    funding = client.fetch_funding_rates(symbol, start, end)
    print(f"Funding ok: {len(funding)} row(s)")

    mark = client.fetch_mark_price_klines(symbol, "1h", start, end)
    print(f"Mark-price klines ok: {len(mark)} candle(s)")

    premium = client.fetch_premium_index_klines(symbol, "1h", start, end)
    print(f"Premium-index klines ok: {len(premium)} candle(s)")

    request = MarketDataRequest(
        datasets=frozenset(
            {
                DataRequirement.OHLCV,
                DataRequirement.FUNDING_RATES,
                DataRequirement.MARK_PRICE_KLINES,
                DataRequirement.PREMIUM_INDEX_KLINES,
            }
        ),
        ohlcv_interval="1h",
        poll_ohlcv_interval="15m",
    )
    context = prepare_market_context(
        symbols=[symbol],
        start=end - timedelta(hours=24),
        end=end,
        client=client,
        request=request,
        warmup_bars=100,
    )
    symbol_context = context.for_symbol(symbol)
    print(f"Prepared context ok: {len(symbol_context.frame)} frame row(s)")
    print(f"Poll candles ok: {len(context.slice_poll_candles(symbol, context.fetch_start, end))} row(s)")
    print("Bybit data smoke test completed.")


if __name__ == "__main__":
    main()
