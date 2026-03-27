from __future__ import annotations

import random
import signal
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from backtester.models import Signal
from backtester.preview import floor_boundary, interval_to_seconds
from marketdata import MarketDataRequest

from .auth_client import BinanceFuturesClient, LiveMarketClient
from .executor import OrderExecutor
from .models import LiveConfig, PositionStatus
from .signal_generator import SignalGenerator
from .tracker import PositionTracker

_INTENSIVE_POLL_LEAD_SECONDS = 120.0
_PRE_POLL_LEAD_SECONDS = 10.0


class LiveEngine:
    """Main loop: poll signal generator → execute → track positions."""

    def __init__(
        self,
        generator: SignalGenerator,
        config: LiveConfig | None = None,
    ) -> None:
        self._config = config or LiveConfig.load()
        self._generator = generator
        self._running = False

        # Two-phase interval-aware polling state
        self._market_request = MarketDataRequest.ohlcv_only()
        self._poll_interval = "1h"
        self._poll_interval_seconds = float(interval_to_seconds("1h"))
        self._last_pre_poll_boundary: datetime | None = None
        self._last_signal_poll_boundary: datetime | None = None
        self._pre_poll_balance: float | None = None  # set by pre-poll, read by signal poll

        # Built during start()
        self._futures_client: BinanceFuturesClient | None = None
        self._market_client: LiveMarketClient | None = None
        self._executor: OrderExecutor | None = None
        self._tracker: PositionTracker | None = None

    def start(self) -> None:
        """Run the live trading loop until interrupted."""
        self._futures_client = BinanceFuturesClient(self._config)
        self._market_client = LiveMarketClient()
        self._executor = OrderExecutor(self._futures_client, self._config)
        self._tracker = PositionTracker(self._futures_client, self._executor, self._config)
        self._tracker.load_state()
        self._tracker.reconcile_with_exchange()

        market_request = self._generator.market_data_request()
        self._configure_market_schedule(market_request)

        # Setup signal generator
        self._generator.setup(self._market_client)

        # Handle SIGINT gracefully
        self._running = True
        original_handler = signal.getsignal(signal.SIGINT)

        def _shutdown(signum: int, frame: Any) -> None:
            print("\nShutting down…", file=sys.stderr)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)

        print(
            f"LiveEngine started | "
            f"size={self._config.position_size_usdt} USDT | "
            f"max_positions={self._config.max_concurrent_positions} | "
            f"hold_max={self._config.max_holding_hours}h | "
            f"analysis_interval={market_request.ohlcv_interval} | "
            f"poll_interval={market_request.effective_poll_ohlcv_interval} | "
            f"market_data={','.join(sorted(req.value for req in market_request.datasets))} | "
            f"testnet={self._config.testnet}",
            file=sys.stderr,
        )

        # Print initial available capital
        try:
            balance = self._futures_client.get_available_balance()
            print(f"Available capital: {balance:.2f} USDT", file=sys.stderr)
        except Exception as exc:
            print(f"Could not fetch balance: {exc}", file=sys.stderr)

        try:
            self._loop()
        finally:
            if self._tracker is not None:
                self._tracker.save_state(force=True)
            self._generator.teardown()
            signal.signal(signal.SIGINT, original_handler)
            print("LiveEngine stopped.", file=sys.stderr)

    # -- Main loop -------------------------------------------------------------

    def _loop(self) -> None:
        assert self._tracker is not None
        assert self._executor is not None
        assert self._futures_client is not None

        check_interval = self._config.order_check_interval_seconds

        while self._running:
            now_utc = self._futures_client.server_now()

            # 1. Check for fills on existing positions
            self._tracker.check_fills(now_utc)
            self._tracker.save_state()

            # 2. Pre-poll: capital check at ~xx:59:50
            if self._should_pre_poll(now_utc):
                self._do_pre_poll(now_utc)

            # 3. Signal poll: check signals at xx:00:01
            if self._should_signal_poll(now_utc, check_interval):
                self._do_signal_poll(now_utc)
                self._tracker.save_state()

            # 4. Sleep until next check
            time.sleep(self._sleep_interval_seconds(now_utc, check_interval))

    # -- Interval-aware signal polling ------------------------------------------

    def _configure_market_schedule(self, market_request: MarketDataRequest) -> None:
        self._market_request = market_request
        self._poll_interval = market_request.effective_poll_ohlcv_interval
        self._poll_interval_seconds = float(
            interval_to_seconds(market_request.effective_poll_ohlcv_interval)
        )

    def _current_poll_boundary(self, now_utc: datetime) -> datetime:
        return floor_boundary(now_utc, self._poll_interval)

    def _next_poll_boundary(self, now_utc: datetime) -> datetime:
        return self._current_poll_boundary(now_utc) + timedelta(seconds=self._poll_interval_seconds)

    def _intensive_poll_lead_seconds(self, check_interval: float) -> float:
        return min(
            _INTENSIVE_POLL_LEAD_SECONDS,
            max(check_interval, self._poll_interval_seconds / 6.0),
        )

    def _pre_poll_lead_seconds(self, check_interval: float) -> float:
        return min(
            _PRE_POLL_LEAD_SECONDS,
            max(check_interval, self._poll_interval_seconds / 6.0),
        )

    def _should_pre_poll(self, now_utc: datetime) -> bool:
        """True once per poll interval shortly before the next poll boundary."""
        next_boundary = self._next_poll_boundary(now_utc)
        lead = self._pre_poll_lead_seconds(self._config.order_check_interval_seconds)
        if next_boundary - timedelta(seconds=lead) <= now_utc < next_boundary:
            return self._last_pre_poll_boundary != next_boundary
        return False

    def _should_signal_poll(self, now_utc: datetime, check_interval: float) -> bool:
        """True once per poll interval immediately after the boundary."""
        boundary = self._current_poll_boundary(now_utc)
        elapsed = (now_utc - boundary).total_seconds()
        if 0.0 <= elapsed <= 1.0 + check_interval:
            return self._last_signal_poll_boundary != boundary
        return False

    def _has_local_active_positions(self) -> bool:
        assert self._tracker is not None
        return any(
            pos.status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN)
            for pos in self._tracker.positions
        )

    def _sleep_interval_seconds(self, now_utc: datetime, check_interval: float) -> float:
        """Use coarse idle polling until the lead-up window before the next poll boundary."""
        if self._has_local_active_positions():
            return check_interval

        seconds_until_boundary = max(
            0.0,
            (self._next_poll_boundary(now_utc) - now_utc).total_seconds(),
        )
        lead = self._intensive_poll_lead_seconds(check_interval)
        if seconds_until_boundary <= lead:
            return check_interval

        coarse_sleep = seconds_until_boundary - lead
        return max(check_interval, coarse_sleep)

    def _configured_leverage(self) -> float:
        leverage = getattr(self._generator, "leverage", 1.0)
        try:
            leverage = float(leverage)
        except (TypeError, ValueError):
            return 1.0
        return leverage if leverage > 0 else 1.0

    def _effective_buying_power(self, available_balance: float) -> float:
        leverage = self._configured_leverage()
        return max(0.0, available_balance) * leverage

    def _affordable_entries(self, available_balance: float) -> int:
        needed = self._config.position_size_usdt
        if needed <= 0:
            return 0
        return int(self._effective_buying_power(available_balance) // needed)

    def _do_pre_poll(self, now_utc: datetime) -> None:
        """Check available capital shortly before the next signal poll boundary."""
        assert self._futures_client is not None
        assert self._tracker is not None

        self._tracker.reconcile_with_exchange()
        next_boundary = self._next_poll_boundary(now_utc)
        self._last_pre_poll_boundary = next_boundary

        open_count = self._tracker.open_count
        open_slots = self._config.max_concurrent_positions - open_count

        print(
            f"\n--- Pre-poll check {now_utc.strftime('%H:%M:%S')} UTC "
            f"(next signal poll at {next_boundary.strftime('%H:%M:%S')}) ---",
            file=sys.stderr,
        )
        print(
            f"Open positions: {open_count}/{self._config.max_concurrent_positions} "
            f"| Open slots: {open_slots}",
            file=sys.stderr,
        )

        try:
            balance = self._futures_client.get_available_balance()
            self._pre_poll_balance = balance
            needed = self._config.position_size_usdt
            buying_power = self._effective_buying_power(balance)
            affordable = self._affordable_entries(balance)
            print(
                f"Available balance: {balance:.2f} USDT | "
                f"Leveraged buying power: {buying_power:.2f} USDT "
                f"(leverage={self._configured_leverage():.2f}x) | "
                f"Affordable entries: {affordable}",
                file=sys.stderr,
            )
            if affordable <= 0:
                print(
                    f"Insufficient capital: need {needed:.2f} USDT, "
                    f"leveraged buying power is {buying_power:.2f} USDT — will skip signal poll",
                    file=sys.stderr,
                )
            elif open_slots <= 0:
                print(
                    "No open slots — will skip signal poll",
                    file=sys.stderr,
                )
            else:
                print("Capital ready — will check signals at candle close", file=sys.stderr)
        except Exception as exc:
            self._pre_poll_balance = None
            print(f"Could not fetch balance: {exc}", file=sys.stderr)

    def _do_signal_poll(self, now_utc: datetime) -> None:
        """Check for signals immediately after the strategy poll boundary."""
        assert self._tracker is not None
        assert self._futures_client is not None

        self._tracker.reconcile_with_exchange()
        boundary = self._current_poll_boundary(now_utc)
        self._last_signal_poll_boundary = boundary

        print(
            f"\n--- Signal poll {now_utc.strftime('%H:%M:%S')} UTC "
            f"(boundary {boundary.strftime('%H:%M:%S')}) ---",
            file=sys.stderr,
        )

        # Re-check slots (a fill may have freed one since pre-poll)
        open_slots = self._config.max_concurrent_positions - self._tracker.open_count
        if open_slots <= 0:
            print("Skipping: no open slots", file=sys.stderr)
            return

        # Use pre-poll balance if available, otherwise fetch now (e.g. engine just started)
        needed = self._config.position_size_usdt
        if self._last_pre_poll_boundary == boundary and self._pre_poll_balance is not None:
            balance = self._pre_poll_balance
        else:
            try:
                balance = self._futures_client.get_available_balance()
            except Exception as exc:
                print(f"Could not fetch balance: {exc} — skipping", file=sys.stderr)
                return
        buying_power = self._effective_buying_power(balance)
        affordable = self._affordable_entries(balance)
        print(
            f"Available balance: {balance:.2f} USDT | "
            f"Leveraged buying power: {buying_power:.2f} USDT "
            f"(leverage={self._configured_leverage():.2f}x) | "
            f"Affordable entries: {affordable}",
            file=sys.stderr,
        )

        if affordable <= 0:
            print(
                f"Insufficient capital: need {needed:.2f} USDT, "
                f"leveraged buying power is {buying_power:.2f} USDT — skipping",
                file=sys.stderr,
            )
            return

        # Capital can cover at most this many new positions
        max_entries = min(open_slots, affordable)

        signals = self._poll_generator(now_utc)

        if not signals:
            print("No signals this hour.", file=sys.stderr)
            return

        for sig in signals:
            print(
                f"  Signal: {sig.position_type.value} {sig.ticker}"
                + (f" | strategy={sig.metadata.get('strategy', '?')}" if sig.metadata else ""),
                file=sys.stderr,
            )

        executable_signals: list[Signal] = []
        external_skipped = 0
        for sig in signals:
            if self._tracker.has_external_conflict(sig):
                external_skipped += 1
                print(
                    f"  Skipping {sig.ticker}: existing untracked exchange exposure",
                    file=sys.stderr,
                )
                continue
            executable_signals.append(sig)

        random.shuffle(executable_signals)
        for sig in executable_signals[:max_entries]:
            self._execute_signal(sig)

        skipped = len(executable_signals) - max_entries
        if external_skipped > 0:
            print(f"  {external_skipped} signal(s) skipped (untracked exchange exposure)", file=sys.stderr)
        if skipped > 0:
            reason = "no open slots" if affordable >= open_slots else "insufficient capital"
            print(f"  {skipped} signal(s) skipped ({reason})", file=sys.stderr)

    def _poll_generator(self, now_utc: datetime) -> list[Signal]:
        try:
            self._generator.set_poll_time(now_utc)
            result = self._generator.poll()
        except Exception as exc:
            print(f"Signal generator error: {exc}", file=sys.stderr)
            return []
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]

    def _execute_signal(self, sig: Signal) -> None:
        assert self._executor is not None
        assert self._tracker is not None
        assert self._futures_client is not None
        try:
            position = self._executor.execute_signal(sig)
            self._tracker.add_position(position)
            self._tracker.save_state()
            print(
                f"[{position.position_id}] Signal executed: "
                f"{sig.position_type.value} {sig.ticker} "
                f"qty={position.quantity}",
                file=sys.stderr,
            )
            try:
                balance = self._futures_client.get_available_balance()
                print(f"Available capital: {balance:.2f} USDT", file=sys.stderr)
            except Exception:
                pass
        except Exception as exc:
            print(
                f"Failed to execute signal {sig.ticker}: {exc}",
                file=sys.stderr,
            )
