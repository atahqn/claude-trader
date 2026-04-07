//! Native exit-resolution engine for the backtester.
//!
//! Provides a Rust implementation of the 3-level hierarchical exit resolver
//! (HOUR → MINUTE → TRADE) that dominates CPU time during backtesting.
//! When installed, `backtester.resolver` dispatches here automatically;
//! otherwise it falls back to pure Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ── Internal types ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Candle {
    open_time_ms: i64,
    close_time_ms: i64,
    high: f64,
    low: f64,
}

#[derive(Clone, Copy)]
struct Trade {
    timestamp_ms: i64,
    price: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Long,
    Short,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Barrier {
    Open,      // neither TP nor SL hit
    Tp,
    Sl,
    Ambiguous, // both hit within the same candle
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Hour,
    Minute,
    Trade,
}

struct Resolution {
    is_tp: bool,
    time_ms: i64,
    price: f64,
    level: Level,
    used_fallback: bool,
}

// ── Time helpers ────────────────────────────────────────────────────────

#[inline(always)]
fn floor_hour(ms: i64) -> i64 {
    ms - ms.rem_euclid(3_600_000)
}

#[inline(always)]
fn floor_minute(ms: i64) -> i64 {
    ms - ms.rem_euclid(60_000)
}

// ── Barrier logic ───────────────────────────────────────────────────────

#[inline]
fn barrier_outcome(c: &Candle, side: Side, tp: f64, sl: f64) -> Barrier {
    let (tp_hit, sl_hit) = match side {
        Side::Long => (c.high >= tp, c.low <= sl),
        Side::Short => (c.low <= tp, c.high >= sl),
    };
    match (tp_hit, sl_hit) {
        (false, false) => Barrier::Open,
        (true, false) => Barrier::Tp,
        (false, true) => Barrier::Sl,
        (true, true) => Barrier::Ambiguous,
    }
}

fn barrier_outcome_multi(candles: &[Candle], side: Side, tp: f64, sl: f64) -> Barrier {
    let mut tp_hit = false;
    let mut sl_hit = false;
    for c in candles {
        match side {
            Side::Long => {
                tp_hit = tp_hit || c.high >= tp;
                sl_hit = sl_hit || c.low <= sl;
            }
            Side::Short => {
                tp_hit = tp_hit || c.low <= tp;
                sl_hit = sl_hit || c.high >= sl;
            }
        }
        if tp_hit && sl_hit {
            return Barrier::Ambiguous;
        }
    }
    match (tp_hit, sl_hit) {
        (true, false) => Barrier::Tp,
        (false, true) => Barrier::Sl,
        _ => Barrier::Open,
    }
}

// ── Trade-level resolution ──────────────────────────────────────────────

fn resolve_with_trades(
    trades: &[Trade],
    side: Side,
    tp: f64,
    sl: f64,
    start_ms: i64,
) -> Option<Resolution> {
    for t in trades {
        if t.timestamp_ms < start_ms {
            continue;
        }
        let hit = match side {
            Side::Long => {
                if t.price >= tp {
                    Some(true)
                } else if t.price <= sl {
                    Some(false)
                } else {
                    None
                }
            }
            Side::Short => {
                if t.price <= tp {
                    Some(true)
                } else if t.price >= sl {
                    Some(false)
                } else {
                    None
                }
            }
        };
        if let Some(is_tp) = hit {
            return Some(Resolution {
                is_tp,
                time_ms: t.timestamp_ms,
                price: if is_tp { tp } else { sl },
                level: Level::Trade,
                used_fallback: false,
            });
        }
    }
    None
}

// ── Minute-level resolution ─────────────────────────────────────────────

fn resolve_candles_minute(
    candles: &[Candle],
    side: Side,
    tp: f64,
    sl: f64,
    fetch_trades: &mut dyn FnMut(i64, i64) -> Vec<Trade>,
) -> Option<Resolution> {
    for c in candles {
        match barrier_outcome(c, side, tp, sl) {
            Barrier::Open => continue,
            Barrier::Tp => {
                return Some(Resolution {
                    is_tp: true,
                    time_ms: c.close_time_ms,
                    price: tp,
                    level: Level::Minute,
                    used_fallback: false,
                });
            }
            Barrier::Sl => {
                return Some(Resolution {
                    is_tp: false,
                    time_ms: c.close_time_ms,
                    price: sl,
                    level: Level::Minute,
                    used_fallback: false,
                });
            }
            Barrier::Ambiguous => {
                let trades = fetch_trades(c.open_time_ms, c.close_time_ms);
                if let Some(r) =
                    resolve_with_trades(&trades, side, tp, sl, c.open_time_ms)
                {
                    return Some(r);
                }
                // Fallback: SL at minute close
                return Some(Resolution {
                    is_tp: false,
                    time_ms: c.close_time_ms,
                    price: sl,
                    level: Level::Minute,
                    used_fallback: true,
                });
            }
        }
    }
    None
}

fn resolve_candles_minute_approx(
    candles: &[Candle],
    side: Side,
    tp: f64,
    sl: f64,
    rng: &mut u64,
) -> Option<Resolution> {
    for c in candles {
        match barrier_outcome(c, side, tp, sl) {
            Barrier::Open => continue,
            Barrier::Tp => {
                return Some(Resolution {
                    is_tp: true,
                    time_ms: c.close_time_ms,
                    price: tp,
                    level: Level::Minute,
                    used_fallback: false,
                });
            }
            Barrier::Sl => {
                return Some(Resolution {
                    is_tp: false,
                    time_ms: c.close_time_ms,
                    price: sl,
                    level: Level::Minute,
                    used_fallback: false,
                });
            }
            Barrier::Ambiguous => {
                let is_tp = xorshift64(rng) % 2 == 0;
                return Some(Resolution {
                    is_tp,
                    time_ms: c.close_time_ms,
                    price: if is_tp { tp } else { sl },
                    level: Level::Minute,
                    used_fallback: false,
                });
            }
        }
    }
    None
}

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ── Hour-level resolution ───────────────────────────────────────────────

fn resolve_hour_interval(
    candles: &[Candle],
    side: Side,
    tp: f64,
    sl: f64,
    close_ms: i64,
    fetch_trades: &mut dyn FnMut(i64, i64) -> Vec<Trade>,
) -> Option<Resolution> {
    match barrier_outcome_multi(candles, side, tp, sl) {
        Barrier::Open => None,
        Barrier::Tp => Some(Resolution {
            is_tp: true,
            time_ms: close_ms,
            price: tp,
            level: Level::Hour,
            used_fallback: false,
        }),
        Barrier::Sl => Some(Resolution {
            is_tp: false,
            time_ms: close_ms,
            price: sl,
            level: Level::Hour,
            used_fallback: false,
        }),
        Barrier::Ambiguous => {
            if let Some(r) =
                resolve_candles_minute(candles, side, tp, sl, fetch_trades)
            {
                return Some(r);
            }
            // Fallback: SL at hour close
            Some(Resolution {
                is_tp: false,
                time_ms: close_ms,
                price: sl,
                level: Level::Hour,
                used_fallback: true,
            })
        }
    }
}

fn resolve_hour_interval_approx(
    candles: &[Candle],
    side: Side,
    tp: f64,
    sl: f64,
    close_ms: i64,
    rng: &mut u64,
) -> Option<Resolution> {
    match barrier_outcome_multi(candles, side, tp, sl) {
        Barrier::Open => None,
        Barrier::Tp => Some(Resolution {
            is_tp: true,
            time_ms: close_ms,
            price: tp,
            level: Level::Hour,
            used_fallback: false,
        }),
        Barrier::Sl => Some(Resolution {
            is_tp: false,
            time_ms: close_ms,
            price: sl,
            level: Level::Hour,
            used_fallback: false,
        }),
        Barrier::Ambiguous => {
            if let Some(r) =
                resolve_candles_minute_approx(candles, side, tp, sl, rng)
            {
                return Some(r);
            }
            Some(Resolution {
                is_tp: false,
                time_ms: close_ms,
                price: sl,
                level: Level::Hour,
                used_fallback: false,
            })
        }
    }
}

// ── Main resolution ─────────────────────────────────────────────────────

fn do_resolve_exit(
    hour_candles: &[Candle],
    side: Side,
    tp: f64,
    sl: f64,
    entry_ms: i64,
    end_ms: Option<i64>,
    fetch_min: &mut dyn FnMut(i64, i64) -> Vec<Candle>,
    fetch_trades: &mut dyn FnMut(i64, i64) -> Vec<Trade>,
) -> Option<Resolution> {
    let first_hour_end = floor_hour(entry_ms) + 3_600_000;
    let entry_min_end = floor_minute(entry_ms) + 60_000;

    // 1. Entry minute — exact trades
    let entry_trades = fetch_trades(entry_ms, entry_min_end);
    if let Some(r) = resolve_with_trades(&entry_trades, side, tp, sl, entry_ms) {
        return Some(r);
    }

    // 2. Remaining minutes in first hour
    let first_mins = fetch_min(entry_min_end, first_hour_end);
    if let Some(r) = resolve_hour_interval(
        &first_mins,
        side,
        tp,
        sl,
        first_hour_end,
        fetch_trades,
    ) {
        return Some(r);
    }

    if matches!(end_ms, Some(e) if e <= first_hour_end) {
        return None;
    }

    // 3. Subsequent full hours
    let final_hour = end_ms.map(floor_hour);
    for c in hour_candles {
        if c.open_time_ms < first_hour_end {
            continue;
        }
        if matches!(final_hour, Some(fh) if c.open_time_ms >= fh) {
            break;
        }
        match barrier_outcome(c, side, tp, sl) {
            Barrier::Open => continue,
            Barrier::Tp => {
                return Some(Resolution {
                    is_tp: true,
                    time_ms: c.close_time_ms,
                    price: tp,
                    level: Level::Hour,
                    used_fallback: false,
                });
            }
            Barrier::Sl => {
                return Some(Resolution {
                    is_tp: false,
                    time_ms: c.close_time_ms,
                    price: sl,
                    level: Level::Hour,
                    used_fallback: false,
                });
            }
            Barrier::Ambiguous => {
                let mins = fetch_min(c.open_time_ms, c.close_time_ms);
                if let Some(r) =
                    resolve_candles_minute(&mins, side, tp, sl, fetch_trades)
                {
                    return Some(r);
                }
            }
        }
    }

    // 4. Trailing partial hour
    if let (Some(end), Some(fh)) = (end_ms, final_hour) {
        if end > fh {
            return resolve_partial(fh, end, side, tp, sl, fetch_min, fetch_trades);
        }
    }
    None
}

fn do_resolve_exit_approx(
    hour_candles: &[Candle],
    side: Side,
    tp: f64,
    sl: f64,
    entry_ms: i64,
    end_ms: Option<i64>,
    fetch_min: &mut dyn FnMut(i64, i64) -> Vec<Candle>,
    rng: &mut u64,
) -> Option<Resolution> {
    let first_hour_end = floor_hour(entry_ms) + 3_600_000;
    // Approximate mode starts from entry minute (includes entry minute)
    let entry_min_start = floor_minute(entry_ms);

    // 1. First hour
    let first_mins = fetch_min(entry_min_start, first_hour_end);
    if let Some(r) = resolve_hour_interval_approx(
        &first_mins,
        side,
        tp,
        sl,
        first_hour_end,
        rng,
    ) {
        return Some(r);
    }

    if matches!(end_ms, Some(e) if e <= first_hour_end) {
        return None;
    }

    // 2. Subsequent full hours
    let final_hour = end_ms.map(floor_hour);
    for c in hour_candles {
        if c.open_time_ms < first_hour_end {
            continue;
        }
        if matches!(final_hour, Some(fh) if c.open_time_ms >= fh) {
            break;
        }
        match barrier_outcome(c, side, tp, sl) {
            Barrier::Open => continue,
            Barrier::Tp => {
                return Some(Resolution {
                    is_tp: true,
                    time_ms: c.close_time_ms,
                    price: tp,
                    level: Level::Hour,
                    used_fallback: false,
                });
            }
            Barrier::Sl => {
                return Some(Resolution {
                    is_tp: false,
                    time_ms: c.close_time_ms,
                    price: sl,
                    level: Level::Hour,
                    used_fallback: false,
                });
            }
            Barrier::Ambiguous => {
                let mins = fetch_min(c.open_time_ms, c.close_time_ms);
                if let Some(r) =
                    resolve_candles_minute_approx(&mins, side, tp, sl, rng)
                {
                    return Some(r);
                }
            }
        }
    }

    // 3. Trailing partial hour
    if let (Some(end), Some(fh)) = (end_ms, final_hour) {
        if end > fh {
            let full_min_end = floor_minute(end);
            if fh < full_min_end {
                let mins = fetch_min(fh, full_min_end);
                if let Some(r) =
                    resolve_candles_minute_approx(&mins, side, tp, sl, rng)
                {
                    return Some(r);
                }
            }
        }
    }
    None
}

fn resolve_partial(
    start: i64,
    end: i64,
    side: Side,
    tp: f64,
    sl: f64,
    fetch_min: &mut dyn FnMut(i64, i64) -> Vec<Candle>,
    fetch_trades: &mut dyn FnMut(i64, i64) -> Vec<Trade>,
) -> Option<Resolution> {
    if start >= end {
        return None;
    }
    let full_min_end = floor_minute(end);
    if start < full_min_end {
        let mins = fetch_min(start, full_min_end);
        if let Some(r) = resolve_candles_minute(&mins, side, tp, sl, fetch_trades) {
            return Some(r);
        }
    }
    if full_min_end < end {
        let trades = fetch_trades(full_min_end, end);
        return resolve_with_trades(&trades, side, tp, sl, full_min_end);
    }
    None
}

// ── PyO3 helpers ────────────────────────────────────────────────────────

fn to_candles(py: Python<'_>, obj: &Py<PyAny>) -> Vec<Candle> {
    obj.extract::<Vec<(i64, i64, f64, f64)>>(py)
        .unwrap_or_default()
        .into_iter()
        .map(|(ot, ct, h, l)| Candle {
            open_time_ms: ot,
            close_time_ms: ct,
            high: h,
            low: l,
        })
        .collect()
}

fn to_trades(py: Python<'_>, obj: &Py<PyAny>) -> Vec<Trade> {
    obj.extract::<Vec<(i64, f64)>>(py)
        .unwrap_or_default()
        .into_iter()
        .map(|(ts, p)| Trade {
            timestamp_ms: ts,
            price: p,
        })
        .collect()
}

// ── Exported Python functions ───────────────────────────────────────────

/// Hierarchical exit resolution: HOUR → MINUTE → TRADE.
///
/// Arguments use millisecond timestamps instead of datetime objects.
/// The Python wrapper in `backtester.resolver` handles conversion.
#[pyfunction]
#[pyo3(signature = (
    hour_candles, is_long, tp_price, sl_price, entry_time_ms,
    minute_fetcher, agg_trade_fetcher,
    end_time_ms=None, approximate=false, seed=None
))]
fn resolve_exit_rs(
    py: Python<'_>,
    hour_candles: Vec<(i64, i64, f64, f64)>,
    is_long: bool,
    tp_price: f64,
    sl_price: f64,
    entry_time_ms: i64,
    minute_fetcher: PyObject,
    agg_trade_fetcher: PyObject,
    end_time_ms: Option<i64>,
    approximate: bool,
    seed: Option<u64>,
) -> PyResult<Option<(String, i64, f64, String, bool)>> {
    let candles: Vec<Candle> = hour_candles
        .into_iter()
        .map(|(ot, ct, h, l)| Candle {
            open_time_ms: ot,
            close_time_ms: ct,
            high: h,
            low: l,
        })
        .collect();
    let side = if is_long { Side::Long } else { Side::Short };

    let mut fetch_min = |start: i64, end: i64| -> Vec<Candle> {
        minute_fetcher
            .call1(py, (start, end))
            .map(|obj| to_candles(py, &obj))
            .unwrap_or_default()
    };
    let mut fetch_agg = |start: i64, end: i64| -> Vec<Trade> {
        agg_trade_fetcher
            .call1(py, (start, end))
            .map(|obj| to_trades(py, &obj))
            .unwrap_or_default()
    };

    let result = if approximate {
        let mut rng = seed.unwrap_or(42);
        if rng == 0 {
            rng = 1;
        }
        do_resolve_exit_approx(
            &candles,
            side,
            tp_price,
            sl_price,
            entry_time_ms,
            end_time_ms,
            &mut fetch_min,
            &mut rng,
        )
    } else {
        do_resolve_exit(
            &candles,
            side,
            tp_price,
            sl_price,
            entry_time_ms,
            end_time_ms,
            &mut fetch_min,
            &mut fetch_agg,
        )
    };

    Ok(result.map(|r| {
        (
            if r.is_tp { "TP" } else { "SL" }.to_string(),
            r.time_ms,
            r.price,
            match r.level {
                Level::Hour => "1h",
                Level::Minute => "1m",
                Level::Trade => "trade",
            }
            .to_string(),
            r.used_fallback,
        )
    }))
}

/// Compute TP and SL price levels from percentages with fee offset.
#[pyfunction]
#[pyo3(signature = (
    entry_price, is_long,
    tp_pct=None, sl_pct=None, taker_fee_rate=0.0005,
    tp_price_override=None, sl_price_override=None
))]
fn compute_tp_sl_prices_rs(
    entry_price: f64,
    is_long: bool,
    tp_pct: Option<f64>,
    sl_pct: Option<f64>,
    taker_fee_rate: f64,
    tp_price_override: Option<f64>,
    sl_price_override: Option<f64>,
) -> PyResult<(f64, f64)> {
    let tp_price = match tp_price_override {
        Some(p) => p,
        None => {
            let pct = tp_pct.ok_or_else(|| {
                PyValueError::new_err("tp_pct required when tp_price_override is not set")
            })?;
            let fee = taker_fee_rate * 2.0 * 100.0;
            let with_fees = pct + fee;
            if is_long {
                entry_price * (1.0 + with_fees / 100.0)
            } else {
                entry_price * (1.0 - with_fees / 100.0)
            }
        }
    };
    let sl_price = match sl_price_override {
        Some(p) => p,
        None => {
            let pct = sl_pct.ok_or_else(|| {
                PyValueError::new_err("sl_pct required when sl_price_override is not set")
            })?;
            let fee = taker_fee_rate * 2.0 * 100.0;
            let with_fees = (pct - fee).max(0.01);
            if is_long {
                entry_price * (1.0 - with_fees / 100.0)
            } else {
                entry_price * (1.0 + with_fees / 100.0)
            }
        }
    };
    Ok((tp_price, sl_price))
}

/// Return (net_pnl_pct, gross_pnl_pct, fee_drag_pct).
#[pyfunction]
#[pyo3(signature = (entry_price, exit_price, is_long, leverage=1.0, taker_fee_rate=0.0005))]
fn compute_pnl_rs(
    entry_price: f64,
    exit_price: f64,
    is_long: bool,
    leverage: f64,
    taker_fee_rate: f64,
) -> (f64, f64, f64) {
    let gross = if is_long {
        ((exit_price - entry_price) / entry_price) * 100.0 * leverage
    } else {
        ((entry_price - exit_price) / entry_price) * 100.0 * leverage
    };
    let fee_drag = 2.0 * taker_fee_rate * leverage * 100.0;
    (gross - fee_drag, gross, fee_drag)
}

// ── Module registration ─────────────────────────────────────────────────

#[pymodule]
fn resolver_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resolve_exit_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_tp_sl_prices_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pnl_rs, m)?)?;
    Ok(())
}
