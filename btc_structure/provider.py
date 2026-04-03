from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from .config import BtcStructureConfig
from .engine import StructureArtifacts, StructureCheckpoint, simulate_btc_structure
from .features import StructureExperimentResult, StructureLabArtifacts, run_structure_feature_lab

if TYPE_CHECKING:
    from backtester.data import BinanceClient

# BTCUSDT perpetual listing date on Binance
_LISTING_START = datetime(2019, 9, 8, tzinfo=UTC)


def _candles_to_frame(candles: list) -> pd.DataFrame:
    rows = [
        {
            "close_time": c.close_time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df.sort_values("close_time").drop_duplicates("close_time", keep="last").reset_index(drop=True)


class DailyStructureProvider:
    """Fetch, compute, cache, and serve daily BTC structure features.

    Three access patterns:

    Backtesting::

        # In generate_backtest_signals(ctx, symbols, start, end):
        data_cutoff = ctx.for_symbol("BTC/USDT").frame["close_time"].max()
        self._structure.ensure_computed_until(data_cutoff)
        frame = self._structure.merge_onto(frame, COLS, cutoff=data_cutoff)

    Validation-safe: ``cutoff`` ensures the validator's truncated context
    also restricts which daily features are visible — the same feature
    matrix is sliced rather than recomputed, which is sound because the
    engine is strictly causal.

    Live::

        self._structure.refresh_if_stale(self.current_time())
        daily = self._structure.latest()
    """

    def __init__(
        self,
        client: "BinanceClient | None" = None,
        *,
        columns: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        columns:
            Which structure feature columns this provider should compute.
            Only the fib scopes needed for these columns are evaluated.
            Pass ``None`` (default) to compute everything.

            Typical::

                provider = DailyStructureProvider(columns=STRUCTURE_REGIME + STRUCTURE_LEVELS)
        """
        if client is None:
            from backtester.data import BinanceClient
            from backtester.models import MarketType

            client = BinanceClient(market_type=MarketType.FUTURES)
        self._client = client
        self._columns = columns
        self._features: pd.DataFrame | None = None
        self._result: StructureExperimentResult | None = None
        self._computed_until: datetime | None = None
        self._checkpoint: StructureCheckpoint | None = None
        self._cached_frame: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute(self, cutoff: datetime) -> None:
        if self._cached_frame is not None and self._computed_until is not None:
            # Incremental: only fetch candles after the last computation
            new_candles = self._client.fetch_klines(
                "BTC/USDT", "1d", self._computed_until, cutoff,
            )
            new_closed = [c for c in new_candles if c.close_time <= cutoff]
            if new_closed:
                new_frame = _candles_to_frame(new_closed)
                frame = pd.concat(
                    [self._cached_frame, new_frame], ignore_index=True,
                ).drop_duplicates(
                    "close_time", keep="last",
                ).sort_values("close_time").reset_index(drop=True)
            else:
                frame = self._cached_frame
        else:
            # First compute: full history from listing date
            candles = self._client.fetch_klines("BTC/USDT", "1d", _LISTING_START, cutoff)
            closed = [c for c in candles if c.close_time <= cutoff]
            if not closed:
                return
            frame = _candles_to_frame(closed)
        self._cached_frame = frame
        config = BtcStructureConfig.for_interval("1d")
        structure, self._checkpoint = simulate_btc_structure(
            frame, config, checkpoint=self._checkpoint,
        )
        lab = run_structure_feature_lab(structure, columns=self._columns)
        self._result = StructureExperimentResult(structure=structure, lab=lab)
        self._features = lab.feature_matrix
        self._computed_until = cutoff

    def ensure_computed_until(self, cutoff: datetime) -> None:
        """Recompute only if cutoff extends past what we already have."""
        if self._computed_until is not None and cutoff <= self._computed_until:
            return
        self._compute(cutoff)

    def reset(self) -> None:
        """Discard all cached state including checkpoint."""
        self._features = None
        self._result = None
        self._computed_until = None
        self._checkpoint = None
        self._cached_frame = None

    def refresh_if_stale(self, now: datetime) -> bool:
        """Recompute if a new daily candle has closed since last run.

        ``now`` should come from ``SignalGenerator.current_time()``, not
        ``datetime.now()``, so all generators share a consistent clock.
        Returns True if a fresh computation was performed.
        """
        if self._features is not None and self._computed_until is not None:
            last_close = pd.Timestamp(self._features["close_time"].iloc[-1])
            next_expected_close = last_close + timedelta(days=1)
            if now < next_expected_close:
                return False
        self._compute(now)
        return True

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._features is not None

    def latest(self) -> pd.Series:
        """Most recent completed daily row."""
        if self._features is None:
            raise RuntimeError("DailyStructureProvider has not been computed yet")
        return self._features.iloc[-1]

    def features(
        self,
        columns: list[str] | None = None,
        *,
        cutoff: datetime | None = None,
    ) -> pd.DataFrame:
        """Feature matrix, optionally column-filtered and time-truncated."""
        if self._features is None:
            raise RuntimeError("DailyStructureProvider has not been computed yet")
        fm = self._features
        if cutoff is not None:
            fm = fm[fm["close_time"] <= cutoff]
        if columns is None:
            return fm
        return fm[["close_time"] + columns]

    def merge_onto(
        self,
        frame: pd.DataFrame,
        columns: list[str],
        time_column: str = "close_time",
        *,
        cutoff: datetime | None = None,
    ) -> pd.DataFrame:
        """Attach daily structure features onto an intraday frame.

        Uses ``merge_asof`` with ``direction='backward'`` so each row
        only sees features from the most recent **completed** daily candle.

        Parameters
        ----------
        cutoff:
            If given, only daily features with ``close_time <= cutoff``
            participate in the merge.  Strategies should derive this from
            the ``PreparedMarketContext``'s actual data boundary::

                data_cutoff = ctx.for_symbol("BTC/USDT").frame["close_time"].max()

            so that ``validate_no_lookahead`` can prove the path is causal
            (the truncated context produces a smaller ``data_cutoff``).
        """
        if self._features is None:
            raise RuntimeError("DailyStructureProvider has not been computed yet")
        daily = self._features[["close_time"] + columns].copy()
        if cutoff is not None:
            daily = daily[daily["close_time"] <= cutoff]
        return pd.merge_asof(
            frame.sort_values(time_column),
            daily.sort_values("close_time"),
            left_on=time_column,
            right_on="close_time",
            direction="backward",
            suffixes=("", "_daily"),
        )

    @property
    def result(self) -> StructureExperimentResult | None:
        return self._result
