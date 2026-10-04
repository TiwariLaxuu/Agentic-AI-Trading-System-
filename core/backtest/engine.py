"""Vectorized-ish, single-position backtest driven purely by model output
(continuation/reversal probability + confidence) — never by a hard-coded
pattern rule. Entries execute on the NEXT bar's open after a signal (no
same-bar lookahead); positions are held a fixed number of bars or exited
early on an opposing high-confidence signal.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict


def run_backtest(dataset: pd.DataFrame, preds: pd.DataFrame, index: pd.Index,
                  entry_threshold: float = 0.10, min_confidence: float = 0.40,
                  holding_bars: int = 10, cost_bps: float = 2.0,
                  bars_per_year: float = 19656.0) -> BacktestResult:
    close = dataset.loc[index, "price_close"]
    idx = list(index)
    n = len(idx)
    cost = cost_bps / 10000.0

    equity = np.ones(n)
    position = 0  # -1, 0, +1
    entry_price = None
    entry_i = None
    bar_returns = np.zeros(n)
    trades = []

    for i in range(1, n):
        t = idx[i]
        prev_t = idx[i - 1]
        r = close.loc[t] / close.loc[prev_t] - 1

        if position != 0:
            bar_returns[i] = position * r
            held = i - entry_i
            sig = preds.loc[prev_t] if prev_t in preds.index else None
            opposing = sig is not None and (
                (position > 0 and sig["p_down"] - sig["p_up"] > entry_threshold and sig["confidence"] >= min_confidence) or
                (position < 0 and sig["p_up"] - sig["p_down"] > entry_threshold and sig["confidence"] >= min_confidence)
            )
            if held >= holding_bars or opposing:
                exit_price = close.loc[t]
                pnl_pct = position * (exit_price / entry_price - 1) - 2 * cost
                trades.append({
                    "entry_time": idx[entry_i], "exit_time": t, "direction": "long" if position > 0 else "short",
                    "entry_price": entry_price, "exit_price": exit_price, "bars_held": held, "pnl_pct": pnl_pct * 100,
                })
                bar_returns[i] -= cost
                position = 0
                entry_price = None
                entry_i = None
        else:
            sig = preds.loc[prev_t] if prev_t in preds.index else None
            if sig is not None and pd.notna(sig.get("p_up")):
                diff = sig["p_up"] - sig["p_down"]
                if diff > entry_threshold and sig["confidence"] >= min_confidence:
                    position = 1
                    entry_price = close.loc[t]
                    entry_i = i
                    bar_returns[i] = -cost
                elif -diff > entry_threshold and sig["confidence"] >= min_confidence:
                    position = -1
                    entry_price = close.loc[t]
                    entry_i = i
                    bar_returns[i] = -cost

        equity[i] = equity[i - 1] * (1 + bar_returns[i])

    equity_curve = pd.Series(equity, index=idx, name="equity")
    trades_df = pd.DataFrame(trades)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    ret_series = pd.Series(bar_returns, index=idx)
    active = ret_series[ret_series != 0]
    sharpe = float(active.mean() / active.std() * np.sqrt(bars_per_year)) if len(active) > 2 and active.std() > 0 else 0.0

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"]
        losses = trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"]
        win_rate = float(len(wins) / len(trades_df))
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
    else:
        win_rate = profit_factor = avg_win = avg_loss = np.nan

    metrics = {
        "total_return_pct": float((equity_curve.iloc[-1] - 1) * 100) if len(equity_curve) else 0.0,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_ratio": sharpe,
        "num_trades": int(len(trades_df)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
    }
    return BacktestResult(equity_curve, trades_df, metrics)
