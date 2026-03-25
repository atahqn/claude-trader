from __future__ import annotations

import random
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

from backtester.models import Signal

from .auth_client import BinanceFuturesClient, LiveMarketClient
from .executor import OrderExecutor
from .models import LiveConfig, PositionStatus
from .signal_generator import SignalGenerator
from .tracker import PositionTracker

_INTENSIVE_POLL_LEAD_SECONDS = 120.0


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

        # Two-phase hourly polling state
        self._last_pre_poll_hour: int | None = None
        self._last_signal_poll_hour: int | None = None
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

    # -- Two-phase hourly polling ----------------------------------------------

    def _should_pre_poll(self, now_utc: datetime) -> bool:
        """True once per hour at ~xx:59:50."""
        if now_utc.minute == 59 and now_utc.second >= 50:
            # The upcoming hour is (current_hour + 1) % 24
            upcoming_hour = (now_utc.hour + 1) % 24
            if self._last_pre_poll_hour != upcoming_hour:
                return True
        return False

    def _should_signal_poll(self, now_utc: datetime, check_interval: float) -> bool:
        """True once per hour at xx:00:01."""
        if now_utc.minute == 0 and now_utc.second <= 1 + check_interval:
            current_hour = now_utc.hour
            if self._last_signal_poll_hour != current_hour:
                return True
        return False

    def _has_local_active_positions(self) -> bool:
        assert self._tracker is not None
        return any(
            pos.status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN)
            for pos in self._tracker.positions
        )

    def _sleep_interval_seconds(self, now_utc: datetime, check_interval: float) -> float:
        """Use coarse idle polling until the last 2 minutes before candle close."""
        if self._has_local_active_positions():
            return check_interval

        seconds_past_hour = (
            now_utc.minute * 60
            + now_utc.second
            + now_utc.microsecond / 1_000_000
        )
        seconds_until_hour = max(0.0, 3600.0 - seconds_past_hour)
        if seconds_until_hour <= _INTENSIVE_POLL_LEAD_SECONDS:
            return check_interval

        coarse_sleep = seconds_until_hour - _INTENSIVE_POLL_LEAD_SECONDS
        return max(check_interval, coarse_sleep)

    def _do_pre_poll(self, now_utc: datetime) -> None:
        """Check available capital before the hourly candle close."""
        assert self._futures_client is not None
        assert self._tracker is not None

        self._tracker.reconcile_with_exchange()
        upcoming_hour = (now_utc.hour + 1) % 24
        self._last_pre_poll_hour = upcoming_hour

        open_count = self._tracker.open_count
        open_slots = self._config.max_concurrent_positions - open_count

        print(
            f"\n--- Pre-poll check {now_utc.strftime('%H:%M:%S')} UTC "
            f"(next candle closes at {upcoming_hour:02d}:00) ---",
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
            print(f"Available capital: {balance:.2f} USDT", file=sys.stderr)
            if balance < needed:
                print(
                    f"Insufficient capital: need {needed:.2f} USDT, "
                    f"have {balance:.2f} USDT — will skip signal poll",
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
        """Check for signals right after the hourly candle closes."""
        assert self._tracker is not None
        assert self._futures_client is not None

        self._tracker.reconcile_with_exchange()
        current_hour = now_utc.hour
        self._last_signal_poll_hour = current_hour

        print(
            f"\n--- Signal poll {now_utc.strftime('%H:%M:%S')} UTC "
            f"(candle {current_hour:02d}:00 closed) ---",
            file=sys.stderr,
        )

        # Re-check slots (a fill may have freed one since pre-poll)
        open_slots = self._config.max_concurrent_positions - self._tracker.open_count
        if open_slots <= 0:
            print("Skipping: no open slots", file=sys.stderr)
            return

        # Use pre-poll balance if available, otherwise fetch now (e.g. engine just started)
        needed = self._config.position_size_usdt
        if self._last_pre_poll_hour == current_hour and self._pre_poll_balance is not None:
            balance = self._pre_poll_balance
        else:
            try:
                balance = self._futures_client.get_available_balance()
            except Exception as exc:
                print(f"Could not fetch balance: {exc} — skipping", file=sys.stderr)
                return
        print(f"Available capital: {balance:.2f} USDT", file=sys.stderr)

        if balance < needed:
            print(
                f"Insufficient capital: need {needed:.2f} USDT — skipping",
                file=sys.stderr,
            )
            return

        # Capital can cover at most this many new positions
        affordable = int(balance // needed)
        max_entries = min(open_slots, affordable)

        signals = self._poll_generator()

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

    def _poll_generator(self) -> list[Signal]:
        try:
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
