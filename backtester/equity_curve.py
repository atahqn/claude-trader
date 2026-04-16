"""Post-processing granular equity curve for resolved backtest trades.

Fetches intermediate candles (default 15m) for each trade's lifetime and
computes a mark-to-market equity curve with configurable granularity.
This module does not modify any backtester logic — it operates purely on
already-resolved TradeResult objects.
"""

from __future__ import annotations

import csv
import sys
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .models import ExitReason, TradeResult
from .resolver import compute_pnl

if TYPE_CHECKING:
    from .data import BinanceClient


@dataclass(slots=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int


def _interval_to_timedelta(interval: str) -> timedelta:
    """Convert a Binance-style interval string to a timedelta."""
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    raise ValueError(f"Unsupported interval unit: {interval!r}")


def compute_granular_equity_curve(
    trades: list[TradeResult],
    client: BinanceClient,
    interval: str = "15m",
    position_size_usdt: float = 10_000.0,
    starting_capital: float = 10_000.0,
    max_workers: int = 8,
) -> list[EquityPoint]:
    """Compute a granular equity curve from resolved trades.

    For each resolved trade, fetches candles at *interval* between entry and
    exit.  At each candle close time, computes the total portfolio value as::

        starting_capital + realised_PnL + unrealised_PnL

    Parameters
    ----------
    trades:
        Full list of TradeResult objects (UNFILLED trades are filtered out).
    client:
        BinanceClient instance (benefits from disk cache on repeated runs).
    interval:
        Candle interval for granularity (e.g. ``"15m"``, ``"5m"``, ``"1h"``).
    position_size_usdt:
        Notional position size in USDT per trade (scaled by ``size_multiplier``).
    starting_capital:
        Starting portfolio value in USDT.
    max_workers:
        Threads for parallel per-symbol candle fetching.
    """
    resolved = [t for t in trades if t.exit_reason is not ExitReason.UNFILLED]
    if not resolved:
        return []

    # Group by symbol → overall time range needed per symbol.
    symbol_ranges: dict[str, tuple[datetime, datetime]] = {}
    for trade in resolved:
        sym = trade.signal.ticker
        if sym not in symbol_ranges:
            symbol_ranges[sym] = (trade.entry_time, trade.exit_time)
        else:
            s, e = symbol_ranges[sym]
            symbol_ranges[sym] = (min(s, trade.entry_time), max(e, trade.exit_time))

    # Fetch candles per symbol in parallel.
    symbol_times: dict[str, list[datetime]] = {}
    symbol_prices: dict[str, list[float]] = {}

    def _fetch(sym: str) -> tuple[str, list[datetime], list[float]]:
        start, end = symbol_ranges[sym]
        candles = client.fetch_klines(sym, interval, start, end)
        candles.sort(key=lambda c: c.close_time)
        times = [c.close_time for c in candles]
        prices = [c.close for c in candles]
        return sym, times, prices

    print(
        f"Fetching {interval} candles for equity curve ({len(symbol_ranges)} symbol(s))...",
        file=sys.stderr,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, sym): sym for sym in symbol_ranges}
        for fut in as_completed(futures):
            sym, times, prices = fut.result()
            symbol_times[sym] = times
            symbol_prices[sym] = prices

    # Build unified timeline from all candle close-times.
    all_times: set[datetime] = set()
    for times in symbol_times.values():
        all_times.update(times)
    timeline = sorted(all_times)
    if not timeline:
        return []

    # Pre-sort helpers.
    resolved.sort(key=lambda t: t.entry_time)
    by_exit = sorted(resolved, key=lambda t: t.exit_time)

    # Walk the timeline.
    points: list[EquityPoint] = []
    realized_cursor = 0
    cumulative_realized = 0.0

    for t in timeline:
        # Accumulate realised PnL for trades closed at or before t.
        while realized_cursor < len(by_exit) and by_exit[realized_cursor].exit_time <= t:
            trade = by_exit[realized_cursor]
            cumulative_realized += (
                position_size_usdt * trade.signal.size_multiplier * (trade.pnl_pct / 100)
            )
            realized_cursor += 1

        # Mark-to-market open positions.
        unrealized = 0.0
        open_count = 0
        for trade in resolved:
            if trade.entry_time > t:
                break
            if trade.exit_time <= t:
                continue
            # Trade is open at t — find latest price.
            times = symbol_times.get(trade.signal.ticker)
            prices = symbol_prices.get(trade.signal.ticker)
            if not times:
                continue
            idx = bisect_right(times, t) - 1
            if idx < 0:
                continue
            net_pnl_pct, _, _ = compute_pnl(
                trade.entry_price,
                prices[idx],
                trade.signal.position_type,
                trade.signal.leverage,
                trade.signal.taker_fee_rate,
            )
            unrealized += position_size_usdt * trade.signal.size_multiplier * (net_pnl_pct / 100)
            open_count += 1

        points.append(EquityPoint(
            timestamp=t,
            equity=starting_capital + cumulative_realized + unrealized,
            realized_pnl=cumulative_realized,
            unrealized_pnl=unrealized,
            open_positions=open_count,
        ))

    return points


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_equity_csv(points: list[EquityPoint], path: Path) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "equity", "realized_pnl", "unrealized_pnl", "open_positions"])
        for p in points:
            writer.writerow([
                p.timestamp.isoformat(),
                round(p.equity, 2),
                round(p.realized_pnl, 2),
                round(p.unrealized_pnl, 2),
                p.open_positions,
            ])
    return path


def plot_equity_curve(
    points: list[EquityPoint],
    output_path: Path,
    title: str = "Equity Curve",
    starting_capital: float = 10_000.0,
    interval: str = "15m",
) -> Path:
    """Create an interactive Plotly equity-curve chart and save as HTML."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    timestamps = [p.timestamp for p in points]
    equities = [p.equity for p in points]

    # Detect time-gaps larger than 2× the interval and insert None to break lines.
    gap_threshold = _interval_to_timedelta(interval) * 2
    plot_ts: list[datetime | None] = []
    plot_eq: list[float | None] = []
    plot_dd: list[float | None] = []
    plot_realized: list[float | None] = []
    plot_unrealized: list[float | None] = []
    plot_open: list[int | None] = []

    peak = equities[0]
    for i, (t, eq) in enumerate(zip(timestamps, equities)):
        # Insert a gap marker when there's a discontinuity.
        if i > 0 and (t - timestamps[i - 1]) > gap_threshold:
            plot_ts.append(None)
            plot_eq.append(None)
            plot_dd.append(None)
            plot_realized.append(None)
            plot_unrealized.append(None)
            plot_open.append(None)
            peak = eq  # reset peak after gap

        peak = max(peak, eq)
        dd_pct = ((eq - peak) / peak) * 100 if peak > 0 else 0.0

        plot_ts.append(t)
        plot_eq.append(eq)
        plot_dd.append(dd_pct)
        plot_realized.append(points[i].realized_pnl)
        plot_unrealized.append(points[i].unrealized_pnl)
        plot_open.append(points[i].open_positions)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25],
        subplot_titles=("Portfolio Value (USDT)", "Drawdown (%)"),
    )

    # Equity line.
    fig.add_trace(
        go.Scatter(
            x=plot_ts,
            y=plot_eq,
            mode="lines",
            name="Equity",
            line=dict(color="#2196F3", width=1.5),
            customdata=list(zip(plot_realized, plot_unrealized, plot_open)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Equity: $%{y:,.2f}<br>"
                "Realized: $%{customdata[0]:,.2f}<br>"
                "Unrealized: $%{customdata[1]:,.2f}<br>"
                "Open: %{customdata[2]}<extra></extra>"
            ),
            connectgaps=False,
        ),
        row=1, col=1,
    )

    # Starting capital reference.
    fig.add_hline(
        y=starting_capital, line_dash="dash", line_color="gray",
        opacity=0.6, annotation_text="Starting Capital",
        annotation_position="top left", row=1, col=1,
    )

    # Drawdown fill.
    fig.add_trace(
        go.Scatter(
            x=plot_ts,
            y=plot_dd,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="crimson", width=1),
            fillcolor="rgba(220, 20, 60, 0.2)",
            hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
            connectgaps=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    output_path = Path(output_path)
    fig.write_html(str(output_path))
    return output_path
