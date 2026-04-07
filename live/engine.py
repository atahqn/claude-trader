from __future__ import annotations

import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from backtester.models import Signal
from backtester.preview import floor_boundary, interval_to_seconds
from marketdata import MarketDataRequest

from .auth_client import BinanceFuturesClient, LiveMarketClient
from .executor import ExecutionResult, OrderExecutor
from .models import GeneratorBudget, LiveConfig, PositionStatus
from .signal_generator import FatalSignalError, SignalGenerator
from .tracker import PositionTracker

_INTENSIVE_POLL_LEAD_SECONDS = 120.0
_PRE_POLL_LEAD_SECONDS = 10.0
_OPEN_FILL_CHECK_SECONDS = 30.0


@dataclass
class _GeneratorSlot:
    """Per-generator scheduling and budget state."""

    generator: SignalGenerator
    budget: GeneratorBudget
    strategy_id: str
    declared_symbols: frozenset[str]
    poll_interval: str
    poll_interval_seconds: float
    last_pre_poll_boundary: datetime | None = None
    last_signal_poll_boundary: datetime | None = None
    last_calibration_time: datetime | None = None


class LiveEngine:
    """Main loop: poll signal generators -> execute -> track positions.

    Supports a single generator (backward-compatible) or multiple generators
    with independent poll intervals, symbol spaces, and budgets.
    """

    def __init__(
        self,
        generator: SignalGenerator | None = None,
        config: LiveConfig | None = None,
        *,
        generators: list[tuple[SignalGenerator, GeneratorBudget]] | None = None,
    ) -> None:
        if generator is not None and generators is not None:
            raise ValueError("pass either 'generator' or 'generators', not both")
        if generator is None and generators is None:
            raise ValueError("pass either 'generator' or 'generators'")

        self._config = config or LiveConfig.load()
        self._running = False

        if generator is not None:
            budget = GeneratorBudget(
                position_size_usdt=self._config.position_size_usdt,
                max_positions=self._config.max_concurrent_positions,
            )
            self._generator_inputs: list[tuple[SignalGenerator, GeneratorBudget]] = [
                (generator, budget),
            ]
        else:
            assert generators is not None
            self._generator_inputs = list(generators)

        # Pre-poll shared state
        self._pre_poll_balance: float | None = None
        self._last_reconcile_time: datetime | None = None
        self._last_fill_check_time: datetime | None = None

        # Built during start()
        self._slots: list[_GeneratorSlot] = []
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

        original_handler = signal.getsignal(signal.SIGINT)
        signal_handler_installed = False

        try:
            self._build_slots()
            self._validate_strategy_ids()
            self._validate_symbol_space()

            for slot in self._slots:
                slot.generator.setup(self._market_client)

            # Initial calibration for adaptive strategies
            for slot in self._slots:
                if slot.generator.needs_calibration:
                    self._run_calibration(slot, self._futures_client.server_now())

            self._running = True

            def _shutdown(signum: int, frame: Any) -> None:
                print("\nShutting down…", file=sys.stderr)
                self._running = False

            signal.signal(signal.SIGINT, _shutdown)
            signal_handler_installed = True

            self._print_startup_banner()
            self._loop()
        except FatalSignalError as exc:
            self._running = False
            print(f"Fatal signal generator error: {exc}", file=sys.stderr)
        finally:
            if self._tracker is not None:
                self._tracker.save_state(force=True)
            for slot in self._slots:
                slot.generator.teardown()
            if signal_handler_installed:
                signal.signal(signal.SIGINT, original_handler)
            print("LiveEngine stopped.", file=sys.stderr)

    # -- Slot construction & validation -----------------------------------------

    def _build_slots(self) -> None:
        from backtester.calibration import validate_calibration_config

        self._slots = []
        for gen, budget in self._generator_inputs:
            if gen.needs_calibration:
                validate_calibration_config(gen)
            market_request = gen.market_data_request()
            poll_interval = market_request.effective_poll_ohlcv_interval
            poll_seconds = float(interval_to_seconds(poll_interval))
            self._slots.append(_GeneratorSlot(
                generator=gen,
                budget=budget,
                strategy_id=gen.strategy_id,
                declared_symbols=frozenset(gen.symbols),
                poll_interval=poll_interval,
                poll_interval_seconds=poll_seconds,
            ))

    def _validate_strategy_ids(self) -> None:
        ids = [slot.strategy_id for slot in self._slots]
        seen: set[str] = set()
        for sid in ids:
            if sid in seen:
                raise ValueError(f"Duplicate strategy_id: '{sid}'")
            seen.add(sid)

    def _validate_symbol_space(self) -> None:
        """Assert all generators have completely disjoint symbol spaces."""
        seen: dict[str, str] = {}
        for slot in self._slots:
            for symbol in slot.declared_symbols:
                if symbol in seen:
                    raise ValueError(
                        f"Symbol space conflict: '{symbol}' claimed by both "
                        f"'{seen[symbol]}' and '{slot.strategy_id}'"
                    )
                seen[symbol] = slot.strategy_id

    def _print_startup_banner(self) -> None:
        print(
            f"LiveEngine started | "
            f"slots={len(self._slots)} | "
            f"global_max_positions={self._config.max_concurrent_positions} | "
            f"testnet={self._config.testnet}",
            file=sys.stderr,
        )
        for slot in self._slots:
            calib_info = ""
            if slot.generator.needs_calibration:
                calib_info = (
                    f" | calibration: every {slot.generator.calibration_interval_hours}h"
                    f" lookback {slot.generator.calibration_lookback_hours}h"
                )
            print(
                f"  Slot '{slot.strategy_id}': "
                f"poll={slot.poll_interval} | "
                f"size={slot.budget.position_size_usdt} USDT | "
                f"max_positions={slot.budget.max_positions} | "
                f"symbols={len(slot.declared_symbols)}{calib_info}",
                file=sys.stderr,
            )
        try:
            assert self._futures_client is not None
            balance = self._futures_client.get_available_balance()
            print(f"Available capital: {balance:.2f} USDT", file=sys.stderr)
        except Exception as exc:
            print(f"Could not fetch balance: {exc}", file=sys.stderr)

    # -- Main loop -------------------------------------------------------------

    def _loop(self) -> None:
        assert self._tracker is not None
        assert self._executor is not None
        assert self._futures_client is not None

        check_interval = self._config.order_check_interval_seconds

        while self._running:
            fills_now_utc = self._futures_client.server_now()

            # 1. Check for fills on existing positions
            if self._should_check_fills(fills_now_utc, check_interval):
                self._tracker.check_fills(fills_now_utc)
                self._tracker.save_state()
                self._last_fill_check_time = fills_now_utc

            # 1.5. Recalibrate adaptive strategies if due
            self._maybe_recalibrate(fills_now_utc)

            # 2. Pre-poll: capital check before upcoming boundaries
            pre_poll_now_utc = self._futures_client.server_now()
            if self._should_pre_poll(pre_poll_now_utc):
                self._do_pre_poll(pre_poll_now_utc)

            # 3. Signal poll: check signals at boundaries
            signal_poll_now_utc = self._futures_client.server_now()
            due_slots = self._due_slots(signal_poll_now_utc, check_interval)
            if due_slots:
                self._do_signal_poll(signal_poll_now_utc, due_slots)
                self._tracker.save_state()

            # 4. Sleep until next check
            sleep_now_utc = self._futures_client.server_now()
            time.sleep(self._sleep_interval_seconds(sleep_now_utc, check_interval))

    # -- Multi-boundary scheduling ---------------------------------------------

    def _next_boundary_for_slot(self, slot: _GeneratorSlot, now_utc: datetime) -> datetime:
        return floor_boundary(now_utc, slot.poll_interval) + timedelta(
            seconds=slot.poll_interval_seconds,
        )

    def _earliest_next_boundary(self, now_utc: datetime) -> datetime:
        return min(
            self._next_boundary_for_slot(slot, now_utc) for slot in self._slots
        )

    def _intensive_poll_lead_seconds(self, check_interval: float) -> float:
        min_poll_seconds = min(slot.poll_interval_seconds for slot in self._slots)
        return min(
            _INTENSIVE_POLL_LEAD_SECONDS,
            max(check_interval, min_poll_seconds / 6.0),
        )

    def _pre_poll_lead_seconds(self, check_interval: float) -> float:
        min_poll_seconds = min(slot.poll_interval_seconds for slot in self._slots)
        return min(
            _PRE_POLL_LEAD_SECONDS,
            max(check_interval, min_poll_seconds / 6.0),
        )

    def _should_pre_poll(self, now_utc: datetime) -> bool:
        """True if any slot has an upcoming boundary within the pre-poll lead window."""
        lead = self._pre_poll_lead_seconds(self._config.order_check_interval_seconds)
        for slot in self._slots:
            next_boundary = self._next_boundary_for_slot(slot, now_utc)
            if next_boundary - timedelta(seconds=lead) <= now_utc < next_boundary:
                if slot.last_pre_poll_boundary != next_boundary:
                    return True
        return False

    def _due_slots(
        self, now_utc: datetime, check_interval: float,
    ) -> list[_GeneratorSlot]:
        """Return slots whose poll boundary has just passed."""
        due: list[_GeneratorSlot] = []
        for slot in self._slots:
            boundary = floor_boundary(now_utc, slot.poll_interval)
            elapsed = (now_utc - boundary).total_seconds()
            if 0.0 <= elapsed <= 1.0 + check_interval:
                if slot.last_signal_poll_boundary != boundary:
                    due.append(slot)
        return due

    def _has_local_active_positions(self) -> bool:
        assert self._tracker is not None
        return any(
            pos.status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN)
            for pos in self._tracker.positions
        )

    def _should_check_fills(self, now_utc: datetime, check_interval: float) -> bool:
        """Throttle fill checks: 5s for PENDING_ENTRY, 30s for OPEN."""
        if not any(
            p.status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN)
            for p in self._tracker.positions
        ):
            return False
        if self._last_fill_check_time is None:
            return True
        has_pending = any(
            p.status is PositionStatus.PENDING_ENTRY
            for p in self._tracker.positions
        )
        min_interval = check_interval if has_pending else _OPEN_FILL_CHECK_SECONDS
        return (now_utc - self._last_fill_check_time).total_seconds() >= min_interval

    def _fill_check_interval(self, check_interval: float) -> float:
        """Return the appropriate fill-check interval for current positions."""
        has_pending = any(
            p.status is PositionStatus.PENDING_ENTRY
            for p in self._tracker.positions
        )
        return check_interval if has_pending else _OPEN_FILL_CHECK_SECONDS

    def _sleep_interval_seconds(self, now_utc: datetime, check_interval: float) -> float:
        """Smart sleep: coarse when idle, intensive near boundaries."""
        seconds_until_boundary = max(
            0.0,
            (self._earliest_next_boundary(now_utc) - now_utc).total_seconds(),
        )

        if self._has_local_active_positions():
            return min(self._fill_check_interval(check_interval), seconds_until_boundary)

        lead = self._intensive_poll_lead_seconds(check_interval)
        if seconds_until_boundary <= lead:
            return min(check_interval, seconds_until_boundary)

        coarse_sleep = seconds_until_boundary - lead
        return max(check_interval, coarse_sleep)

    # -- Adaptive calibration ---------------------------------------------------

    def _maybe_recalibrate(self, now_utc: datetime) -> None:
        """Recalibrate adaptive strategies whose interval has elapsed."""
        for slot in self._slots:
            if not slot.generator.needs_calibration:
                continue
            if slot.last_calibration_time is None:
                # Initial calibration should have run in start(); skip here.
                continue
            interval = timedelta(hours=slot.generator.calibration_interval_hours)
            if now_utc - slot.last_calibration_time >= interval:
                self._run_calibration(slot, now_utc)

    def _run_calibration(self, slot: _GeneratorSlot, now_utc: datetime) -> None:
        """Fetch lookback data, search parameters, and set active_params."""
        import pandas as pd

        from backtester.calibration import search_parameters
        from live.signal_generator import CompositeSignalGenerator

        gen = slot.generator
        lookback_hours = gen.calibration_lookback_hours
        # For composites, use the max lookback across calibrating children.
        if isinstance(gen, CompositeSignalGenerator):
            children = gen.calibration_children()
            if not children:
                return
            lookback_hours = max(c.calibration_lookback_hours for c in children)

        lookback_start = now_utc - timedelta(hours=lookback_hours)
        assert self._market_client is not None

        try:
            frames: list[pd.DataFrame] = []
            for symbol in gen.symbols:
                candles = self._market_client.fetch_klines(
                    symbol=symbol.replace("/", ""),
                    interval=gen.analysis_interval,
                    start=lookback_start,
                    end=now_utc,
                )
                if candles:
                    # Exclude in-progress candles: only completed bars.
                    completed = [c for c in candles if c.close_time < now_utc]
                    if not completed:
                        continue
                    rows = [
                        {
                            "open_time": c.open_time,
                            "close_time": c.close_time,
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume,
                            "taker_buy_volume": c.taker_buy_volume,
                            "symbol": symbol,
                        }
                        for c in completed
                    ]
                    frames.append(pd.DataFrame(rows))

            if not frames:
                print(
                    f"Calibration skipped for {slot.strategy_id}: no lookback data",
                    file=sys.stderr,
                )
                return

            raw_frame = pd.concat(frames, ignore_index=True)

            # Calibrate each child independently for composites.
            if isinstance(gen, CompositeSignalGenerator):
                any_success = False
                for child in gen.calibration_children():
                    child_symbols = set(child.symbols)
                    child_frame = raw_frame[raw_frame["symbol"].isin(child_symbols)]
                    if child_frame.empty:
                        continue
                    if self._calibrate_single_live(
                        child, child_frame, now_utc, slot.strategy_id,
                    ):
                        any_success = True
                if any_success:
                    slot.last_calibration_time = now_utc
            else:
                if self._calibrate_single_live(gen, raw_frame, now_utc, slot.strategy_id):
                    slot.last_calibration_time = now_utc

        except Exception as exc:
            print(
                f"Calibration failed for {slot.strategy_id}: {exc}",
                file=sys.stderr,
            )

    def _calibrate_single_live(
        self,
        gen: SignalGenerator,
        raw_frame: Any,
        now_utc: datetime,
        slot_id: str,
    ) -> bool:
        """Calibrate a single generator. Returns True on success."""
        from backtester.calibration import search_parameters

        frame = gen.build_calibration_frame(raw_frame, now_utc)
        if frame is None or frame.empty:
            print(
                f"Calibration skipped for {slot_id}/{gen.strategy_id}: "
                f"build_calibration_frame returned empty",
                file=sys.stderr,
            )
            return False

        result = search_parameters(
            param_space=gen.param_space(),
            score_fn=gen.score_params,
            frame=frame,
        )

        if result is not None:
            gen.active_params = result.best_params
            print(
                f"Calibrated {slot_id}/{gen.strategy_id}: "
                f"params={result.best_params} "
                f"score={result.best_score:.4f} "
                f"({result.candidates_evaluated} candidates)",
                file=sys.stderr,
            )
            return True

        print(
            f"Calibration returned no result for {slot_id}/{gen.strategy_id}",
            file=sys.stderr,
        )
        return False

    # -- Pre-poll: shared reconcile + balance ----------------------------------

    def _do_pre_poll(self, now_utc: datetime) -> None:
        assert self._futures_client is not None
        assert self._tracker is not None

        # One reconcile for the whole engine
        self._tracker.reconcile_with_exchange()
        self._last_reconcile_time = now_utc

        # Mark each due slot's pre-poll boundary
        lead = self._pre_poll_lead_seconds(self._config.order_check_interval_seconds)
        for slot in self._slots:
            next_boundary = self._next_boundary_for_slot(slot, now_utc)
            if next_boundary - timedelta(seconds=lead) <= now_utc < next_boundary:
                slot.last_pre_poll_boundary = next_boundary

        # One balance fetch
        try:
            balance = self._futures_client.get_available_balance()
            self._pre_poll_balance = balance
        except Exception as exc:
            self._pre_poll_balance = None
            print(f"Could not fetch balance: {exc}", file=sys.stderr)
            return

        global_open = self._tracker.open_count
        print(
            f"\n--- Pre-poll check {now_utc.strftime('%H:%M:%S')} UTC ---\n"
            f"Global open: {global_open}/{self._config.max_concurrent_positions} | "
            f"Balance: {balance:.2f} USDT",
            file=sys.stderr,
        )
        for slot in self._slots:
            slot_open = self._tracker.open_count_for(slot.strategy_id)
            print(
                f"  {slot.strategy_id}: {slot_open}/{slot.budget.max_positions} positions",
                file=sys.stderr,
            )

    # -- Signal poll: parallel dispatch + capital allocation -------------------

    def _do_signal_poll(
        self, now_utc: datetime, due_slots: list[_GeneratorSlot],
    ) -> None:
        assert self._tracker is not None
        assert self._futures_client is not None

        # Reconcile if not done recently in pre-poll
        if (
            self._last_reconcile_time is None
            or (now_utc - self._last_reconcile_time).total_seconds() > 5.0
        ):
            self._tracker.reconcile_with_exchange()
            self._last_reconcile_time = now_utc

        print(
            f"\n--- Signal poll {now_utc.strftime('%H:%M:%S')} UTC | "
            f"slots due: {', '.join(s.strategy_id for s in due_slots)} ---",
            file=sys.stderr,
        )

        # Get balance (use pre-poll cache if available)
        balance = self._pre_poll_balance
        if balance is None:
            try:
                balance = self._futures_client.get_available_balance()
            except Exception as exc:
                print(f"Could not fetch balance: {exc} — skipping", file=sys.stderr)
                # Mark boundaries so we don't retry immediately
                for slot in due_slots:
                    slot.last_signal_poll_boundary = floor_boundary(
                        now_utc, slot.poll_interval,
                    )
                return

        # Poll all due generators in parallel
        slot_signals: dict[str, list[Signal]] = {}
        if len(due_slots) == 1:
            slot = due_slots[0]
            slot_signals[slot.strategy_id] = self._poll_generator(slot, now_utc)
        else:
            with ThreadPoolExecutor(max_workers=len(due_slots)) as pool:
                future_to_slot = {
                    pool.submit(self._poll_generator, slot, now_utc): slot
                    for slot in due_slots
                }
                for future in as_completed(future_to_slot):
                    slot = future_to_slot[future]
                    try:
                        slot_signals[slot.strategy_id] = future.result()
                    except FatalSignalError:
                        raise
                    except Exception as exc:
                        print(
                            f"Signal generator error ({slot.strategy_id}): {exc}",
                            file=sys.stderr,
                        )
                        slot_signals[slot.strategy_id] = []

        # Mark boundaries
        for slot in due_slots:
            slot.last_signal_poll_boundary = floor_boundary(
                now_utc, slot.poll_interval,
            )

        # Execute with per-generator budgets + global ceiling
        self._execute_slot_signals(due_slots, slot_signals, balance)

    def _poll_generator(
        self, slot: _GeneratorSlot, now_utc: datetime,
    ) -> list[Signal]:
        try:
            slot.generator.set_poll_time(now_utc)
            result = slot.generator.poll()
        except FatalSignalError:
            raise
        except Exception as exc:
            print(
                f"Signal generator error ({slot.strategy_id}): {exc}",
                file=sys.stderr,
            )
            return []
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]

    def _validate_signals(
        self, slot: _GeneratorSlot, signals: list[Signal],
    ) -> list[Signal]:
        """Filter signals against the generator's declared symbol space."""
        valid: list[Signal] = []
        for sig in signals:
            if sig.ticker not in slot.declared_symbols:
                print(
                    f"WARNING: {slot.strategy_id} emitted signal for undeclared "
                    f"symbol '{sig.ticker}' — skipping",
                    file=sys.stderr,
                )
                continue
            valid.append(sig)
        return valid

    # -- Capital allocation ----------------------------------------------------

    def _execute_slot_signals(
        self,
        due_slots: list[_GeneratorSlot],
        slot_signals: dict[str, list[Signal]],
        balance: float,
    ) -> None:
        assert self._tracker is not None

        global_open = self._tracker.open_count
        global_max = self._config.max_concurrent_positions
        global_remaining = global_max - global_open
        remaining_balance = balance

        print(
            f"Available balance: {balance:.2f} USDT | "
            f"Global slots: {global_max - global_open}/{global_max}",
            file=sys.stderr,
        )

        for slot in due_slots:
            raw_signals = slot_signals.get(slot.strategy_id, [])
            if not raw_signals:
                continue

            # Runtime symbol validation
            signals = self._validate_signals(slot, raw_signals)
            if not signals:
                continue

            for sig in signals:
                print(
                    f"  Signal: {sig.position_type.value} {sig.ticker}"
                    + (
                        f" | strategy={sig.metadata.get('strategy', '?')}"
                        if sig.metadata
                        else ""
                    )
                    + f" | slot={slot.strategy_id}",
                    file=sys.stderr,
                )

            # Per-slot budget
            slot_open = self._tracker.open_count_for(slot.strategy_id)
            slot_remaining = slot.budget.max_positions - slot_open

            # Intersect with global remaining
            max_entries = min(slot_remaining, global_remaining)
            if max_entries <= 0:
                reason = "global ceiling" if slot_remaining > 0 else "slot budget"
                print(
                    f"  {slot.strategy_id}: {len(signals)} signal(s) skipped ({reason})",
                    file=sys.stderr,
                )
                continue

            # Further cap by affordable entries
            affordable = self._affordable_entries_for_budget(
                remaining_balance, slot.budget.position_size_usdt, slot.generator,
            )
            max_entries = min(max_entries, affordable)

            if max_entries <= 0:
                print(
                    f"  {slot.strategy_id}: {len(signals)} signal(s) skipped "
                    f"(insufficient capital)",
                    file=sys.stderr,
                )
                continue

            # Filter external conflicts
            executable: list[Signal] = []
            external_skipped = 0
            for sig in signals:
                if self._tracker.has_external_conflict(sig):
                    external_skipped += 1
                    print(
                        f"  Skipping {sig.ticker}: existing untracked exchange exposure",
                        file=sys.stderr,
                    )
                    continue
                executable.append(sig)

            random.shuffle(executable)
            executed = 0
            for sig in executable[:max_entries]:
                result = self._execute_signal(
                    sig,
                    available_balance=remaining_balance,
                    position_size_usdt=slot.budget.position_size_usdt,
                    strategy_id=slot.strategy_id,
                )
                if result is not None:
                    remaining_balance = max(
                        0.0, remaining_balance - result.margin_consumed,
                    )
                    executed += 1
                    global_remaining -= 1

            skipped = len(executable) - executed
            if external_skipped > 0:
                print(
                    f"  {external_skipped} signal(s) skipped "
                    f"(untracked exchange exposure)",
                    file=sys.stderr,
                )
            if skipped > 0:
                reason = (
                    "no open slots"
                    if affordable >= slot_remaining
                    else "insufficient capital"
                )
                print(
                    f"  {slot.strategy_id}: {skipped} signal(s) skipped ({reason})",
                    file=sys.stderr,
                )

        if not any(slot_signals.get(s.strategy_id) for s in due_slots):
            print("No signals this poll.", file=sys.stderr)

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _generator_leverage(generator: SignalGenerator) -> float:
        leverage = getattr(generator, "leverage", 1.0)
        try:
            leverage = float(leverage)
        except (TypeError, ValueError):
            return 1.0
        return leverage if leverage > 0 else 1.0

    def _affordable_entries_for_budget(
        self,
        available_balance: float,
        position_size_usdt: float,
        generator: SignalGenerator,
    ) -> int:
        leverage = self._generator_leverage(generator)
        buying_power = max(0.0, available_balance) * leverage
        if position_size_usdt <= 0:
            return 0
        return int(buying_power // position_size_usdt)

    def _execute_signal(
        self,
        sig: Signal,
        *,
        available_balance: float | None = None,
        position_size_usdt: float | None = None,
        strategy_id: str = "",
    ) -> ExecutionResult | None:
        assert self._executor is not None
        assert self._tracker is not None

        try:
            result = self._executor.execute_signal(
                sig,
                available_balance=available_balance,
                position_size_usdt=position_size_usdt,
            )
        except Exception as exc:
            print(
                f"Failed to execute signal {sig.ticker}: {exc}",
                file=sys.stderr,
            )
            return None

        result.position.strategy_id = strategy_id
        self._tracker.add_position(result.position)
        self._tracker.save_state()
        print(
            f"[{result.position.position_id}] Signal executed: "
            f"{sig.position_type.value} {sig.ticker} "
            f"qty={result.position.quantity}"
            + (f" | slot={strategy_id}" if strategy_id else ""),
            file=sys.stderr,
        )
        return result
