from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType

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
class AccountTrade:
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    time: datetime
    realized_pnl: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    position_side: str = "BOTH"


@dataclass(slots=True, frozen=True)
class ExchangeOrder:
    order_id: str
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
    is_conditional: bool = False


@dataclass(slots=True, frozen=True)
class GeneratorBudget:
    """Per-generator budget for multi-generator engine mode."""
    position_size_usdt: float = 100.0
    max_positions: int = 3


@dataclass(slots=True)
class LivePosition:
    signal: Signal
    position_id: str
    strategy_id: str = ""
    status: PositionStatus = PositionStatus.PENDING_ENTRY
    entry_order: ExchangeOrder | None = None
    tp_order: ExchangeOrder | None = None
    sl_order: ExchangeOrder | None = None
    fill_price: float = 0.0
    quantity: float = 0.0
    opened_at: datetime | None = None
    exit_price: float | None = None
    pnl_pct: float | None = None
    gross_pnl_pct: float | None = None
    fee_drag_pct: float | None = None
    closed_at: datetime | None = None


_CONFIG_DIR = Path.home() / ".claude_trader"
_CONFIG_PATH = _CONFIG_DIR / "live_config.json"

_PROD_BASE_URL = "https://api.bybit.com"
_TESTNET_BASE_URL = "https://api-testnet.bybit.com"
_ALLOWED_CONFIG_KEYS = frozenset({
    "api_key",
    "api_secret",
    "api_key_var",
    "api_secret_var",
    "base_url",
    "position_size_usdt",
    "max_concurrent_positions",
    "order_check_interval_seconds",
    "testnet",
})


def _load_local_keys_module() -> ModuleType | None:
    try:
        return import_module("live.local_keys")
    except ModuleNotFoundError:
        return None


def _resolve_secret_reference(name: str, *, required: bool = True) -> str:
    module = _load_local_keys_module()
    if module is None:
        if required:
            raise FileNotFoundError(
                "live/local_keys.py is required to resolve config secret references"
            )
        return ""
    value = getattr(module, name, "")
    if value:
        return str(value)
    if required:
        raise ValueError(f"Secret variable {name!r} was not found in live/local_keys.py")
    return ""


@dataclass(slots=True, frozen=True)
class LiveConfig:
    api_key: str
    api_secret: str
    base_url: str = _PROD_BASE_URL
    position_size_usdt: float = 100.0
    max_concurrent_positions: int = 3
    order_check_interval_seconds: float = 5.0
    testnet: bool = False

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key must not be empty")
        if not self.api_secret:
            raise ValueError("api_secret must not be empty")
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if self.position_size_usdt <= 0:
            raise ValueError("position_size_usdt must be greater than zero")
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be greater than zero")
        if self.order_check_interval_seconds <= 0:
            raise ValueError("order_check_interval_seconds must be greater than zero")
        if self.testnet and self.base_url == _PROD_BASE_URL:
            object.__setattr__(self, "base_url", _TESTNET_BASE_URL)

    @property
    def is_testnet(self) -> bool:
        return self.base_url == _TESTNET_BASE_URL

    def with_overrides(
        self,
        *,
        use_testnet: bool = False,
        position_size_usdt: float | None = None,
        max_concurrent_positions: int | None = None,
    ) -> LiveConfig:
        return LiveConfig(
            api_key=self.api_key,
            api_secret=self.api_secret,
            base_url=(
                _TESTNET_BASE_URL
                if use_testnet
                else self.base_url
            ),
            position_size_usdt=(
                self.position_size_usdt
                if position_size_usdt is None
                else position_size_usdt
            ),
            max_concurrent_positions=(
                self.max_concurrent_positions
                if max_concurrent_positions is None
                else max_concurrent_positions
            ),
            order_check_interval_seconds=self.order_check_interval_seconds,
            testnet=use_testnet or self.testnet,
        )

    @staticmethod
    def load(config_path: str | Path | None = None) -> LiveConfig:
        """Load config from an explicit file, env vars, or ~/.claude_trader/live_config.json."""
        if config_path is not None:
            path = Path(config_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            return _config_from_data(_load_config_data(path))

        api_key = os.environ.get("BYBIT_API_KEY", "")
        api_secret = os.environ.get("BYBIT_API_SECRET", "")
        if api_key and api_secret:
            use_testnet = os.environ.get("BYBIT_TESTNET", "").lower() in ("1", "true", "yes")
            return LiveConfig(
                api_key=api_key,
                api_secret=api_secret,
                base_url=os.environ.get(
                    "BYBIT_BASE_URL",
                    _TESTNET_BASE_URL if use_testnet else _PROD_BASE_URL,
                ),
                position_size_usdt=float(os.environ.get("BYBIT_POSITION_SIZE", "100")),
                max_concurrent_positions=int(os.environ.get("BYBIT_MAX_POSITIONS", "3")),
                order_check_interval_seconds=float(os.environ.get("BYBIT_ORDER_CHECK_INTERVAL", "5")),
                testnet=use_testnet,
            )

        local_keys_config = _load_local_keys_config()
        if local_keys_config is not None:
            return local_keys_config

        if _CONFIG_PATH.exists():
            return _config_from_data(_load_config_data(_CONFIG_PATH))

        raise FileNotFoundError(
            f"No API credentials found. Set BYBIT_API_KEY/BYBIT_API_SECRET env vars "
            f"or create {_CONFIG_PATH} or live/local_keys.py"
        )


def _load_config_data(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Live config at {path} must be a JSON object")

    unexpected = sorted(set(data) - _ALLOWED_CONFIG_KEYS)
    if unexpected:
        raise ValueError(
            f"Unsupported live config fields in {path}: {', '.join(unexpected)}. "
            f"Allowed fields: {', '.join(sorted(_ALLOWED_CONFIG_KEYS))}."
        )
    return data


def _config_from_data(data: dict[str, object]) -> LiveConfig:
    config_data = dict(data)

    api_key = str(config_data.pop("api_key", "") or "")
    api_secret = str(config_data.pop("api_secret", "") or "")
    api_key_var = str(config_data.pop("api_key_var", "") or "")
    api_secret_var = str(config_data.pop("api_secret_var", "") or "")

    if not api_key and api_key_var:
        api_key = _resolve_secret_reference(api_key_var)
    if not api_secret and api_secret_var:
        api_secret = _resolve_secret_reference(api_secret_var)

    if not api_key or not api_secret:
        raise ValueError(
            "Config must provide api_key/api_secret or api_key_var/api_secret_var"
        )

    return LiveConfig(api_key=api_key, api_secret=api_secret, **config_data)


def _load_local_keys_config() -> LiveConfig | None:
    module = _load_local_keys_module()
    if module is None:
        return None

    testnet = os.environ.get("BYBIT_TESTNET", "").lower() in ("1", "true", "yes")
    if testnet:
        api_key = getattr(module, "TESTNET_BOT_KEY", "")
        api_secret = getattr(module, "TESTNET_BOT_SECRET", "")
    else:
        api_key = getattr(module, "MAINNET_BOT_KEY", "")
        api_secret = getattr(module, "MAINNET_BOT_SECRET", "")

    if not api_key or not api_secret:
        return None

    return LiveConfig(
        api_key=str(api_key),
        api_secret=str(api_secret),
        base_url=os.environ.get("BYBIT_BASE_URL", _PROD_BASE_URL),
        position_size_usdt=float(os.environ.get("BYBIT_POSITION_SIZE", "100")),
        max_concurrent_positions=int(os.environ.get("BYBIT_MAX_POSITIONS", "3")),
        order_check_interval_seconds=float(os.environ.get("BYBIT_ORDER_CHECK_INTERVAL", "5")),
        testnet=testnet,
    )
