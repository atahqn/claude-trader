from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from backtester.models import Signal


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    NEW = "NEW"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class PositionStatus(StrEnum):
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(slots=True, frozen=True)
class ExchangeOrder:
    order_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    stop_price: float
    status: OrderStatus
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    algo_id: int = 0  # non-zero for algo (conditional) orders


@dataclass(slots=True)
class LivePosition:
    signal: Signal
    position_id: str
    status: PositionStatus = PositionStatus.PENDING_ENTRY
    entry_order: ExchangeOrder | None = None
    tp_order: ExchangeOrder | None = None
    sl_order: ExchangeOrder | None = None
    fill_price: float = 0.0
    quantity: float = 0.0
    pnl_pct: float = 0.0
    gross_pnl_pct: float = 0.0
    fee_drag_pct: float = 0.0
    closed_at: datetime | None = None


_CONFIG_DIR = Path.home() / ".claude_trader"
_CONFIG_PATH = _CONFIG_DIR / "live_config.json"

_PROD_BASE_URL = "https://fapi.binance.com"
_TESTNET_BASE_URL = "https://testnet.binancefuture.com"


@dataclass(slots=True, frozen=True)
class LiveConfig:
    api_key: str
    api_secret: str
    base_url: str = _PROD_BASE_URL
    position_size_usdt: float = 100.0
    max_concurrent_positions: int = 3
    poll_interval_seconds: float = 60.0
    order_check_interval_seconds: float = 5.0
    testnet: bool = False

    def __post_init__(self) -> None:
        if self.testnet and self.base_url == _PROD_BASE_URL:
            object.__setattr__(self, "base_url", _TESTNET_BASE_URL)

    @staticmethod
    def load() -> LiveConfig:
        """Load config from env vars or ~/.claude_trader/live_config.json."""
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        if api_key and api_secret:
            return LiveConfig(
                api_key=api_key,
                api_secret=api_secret,
                base_url=os.environ.get("BINANCE_BASE_URL", _PROD_BASE_URL),
                position_size_usdt=float(os.environ.get("BINANCE_POSITION_SIZE", "100")),
                max_concurrent_positions=int(os.environ.get("BINANCE_MAX_POSITIONS", "3")),
                poll_interval_seconds=float(os.environ.get("BINANCE_POLL_INTERVAL", "60")),
                order_check_interval_seconds=float(os.environ.get("BINANCE_ORDER_CHECK_INTERVAL", "5")),
                testnet=os.environ.get("BINANCE_TESTNET", "").lower() in ("1", "true", "yes"),
            )

        if _CONFIG_PATH.exists():
            data = json.loads(_CONFIG_PATH.read_text())
            return LiveConfig(**data)

        raise FileNotFoundError(
            f"No API credentials found. Set BINANCE_API_KEY/BINANCE_API_SECRET env vars "
            f"or create {_CONFIG_PATH}"
        )
