
import os
from dataclasses import dataclass
from typing import Optional

try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None

@dataclass
class AlpacaConfig:
    key_id: str = os.getenv("ALPACA_API_KEY_ID", "")
    secret_key: str = os.getenv("ALPACA_API_SECRET_KEY", "")
    base_url: str = os.getenv("ALPACA_PAPER_BASE", "https://paper-api.alpaca.markets")
    symbol: str = "AAPL"
    qty: int = 1

def get_client(cfg: AlpacaConfig):
    if tradeapi is None:
        raise ImportError("alpaca-trade-api is not installed. pip install alpaca-trade-api")
    return tradeapi.REST(cfg.key_id, cfg.secret_key, cfg.base_url, api_version='v2')

def place_bracket_order(client, symbol: str, qty: int, side: str, entry_price: float,
                        stop_price: float, take_price: float):
    # Market entry with attached OCO bracket
    order = client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='day',
        order_class='bracket',
        stop_loss={'stop_price': round(stop_price, 2)},
        take_profit={'limit_price': round(take_price, 2)}
    )
    return order


import pandas as pd
from dataclasses import dataclass
from .strategy import build_features
from .risk import RiskConfig, position_size_shares

@dataclass
class BacktestConfig:
    initial_equity: float = 100000.0
    risk: RiskConfig = RiskConfig()

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float

def backtest(df_raw: pd.DataFrame, cfg: BacktestConfig = BacktestConfig()) -> dict:
    df = build_features(df_raw)
    df = df.dropna().reset_index(drop=True)

    equity = cfg.initial_equity
    start_of_day_equity = equity
    open_position = None
    trades = []
    daily_loss = 0.0
    last_date = None

    for i, row in df.iterrows():
        date = row["time"].date()
        price_open = row["open"]
        price_high = row["high"]
        price_low = row["low"]
        price_close = row["close"]
        signal = row["signal"]
        atr_val = row["atr14"]

        if last_date is None or date != last_date:
            # New day: reset daily loss cap tracking
            start_of_day_equity = equity
            daily_loss = 0.0
            last_date = date

        # Update open position for stop or take profit
        if open_position is not None:
            entry_price, shares, stop_price, take_price, entry_time = open_position
            exit_now = False
            exit_price = price_close  # default at close

            # Intraday stop or take
            if price_low <= stop_price <= price_high:
                exit_now = True
                exit_price = stop_price
            elif price_low <= take_price <= price_high:
                exit_now = True
                exit_price = take_price

            if exit_now:
                pnl = (exit_price - entry_price) * shares
                equity += pnl
                daily_loss = min(daily_loss, 0.0) + min(pnl, 0.0)
                trades.append(Trade(entry_time, row["time"], entry_price, exit_price, shares, pnl))
                open_position = None

        # Check daily loss cap
        if (start_of_day_equity - equity) / start_of_day_equity >= cfg.risk.daily_loss_cap_pct:
            # Do not enter new trades today
            continue

        # Entry logic: if no position and signal is long
        if open_position is None and signal == 1 and atr_val > 0 and price_open > 0:
            shares = position_size_shares(equity, price_open, atr_val, cfg.risk)
            if shares > 0:
                stop_price = price_open - cfg.risk.atr_stop_mult * atr_val
                take_price = price_open + cfg.risk.atr_take_mult * atr_val
                # Enter at open price with slippage and commission
                entry_fill = price_open * (1 + cfg.risk.slippage_pct)
                equity -= cfg.risk.commission_per_trade
                open_position = (entry_fill, shares, stop_price, take_price, row["time"])

        # If still holding at the end of data, close at last close
        if i == len(df) - 1 and open_position is not None:
            entry_price, shares, stop_price, take_price, entry_time = open_position
            exit_fill = price_close * (1 - cfg.risk.slippage_pct)
            pnl = (exit_fill - entry_price) * shares
            equity += pnl
            trades.append(Trade(entry_time, row["time"], entry_price, exit_fill, shares, pnl))
            open_position = None

    # Compute summary
    pnl_series = pd.Series([t.pnl for t in trades], name="trade_pnl")
    total_return = (equity / cfg.initial_equity) - 1.0
    max_dd = _max_drawdown_from_equity(cfg.initial_equity, [t.pnl for t in trades])
    ann_return, ann_vol, sharpe = _basic_perf(df, cfg.initial_equity, trades)

    summary = {
        "trades": trades,
        "ending_equity": equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe_like": sharpe
    }
    return summary

def _equity_curve(initial_equity: float, trades):
    eq = [initial_equity]
    for t in trades:
        eq.append(eq[-1] + t.pnl)
    return pd.Series(eq)

def _max_drawdown_from_equity(initial_equity: float, pnls):
    eq = _equity_curve(initial_equity, [type("T", (), {"pnl": p}) for p in pnls])
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())

def _basic_perf(df: pd.DataFrame, initial_equity: float, trades):
    import numpy as np
    if len(trades) == 0:
        return 0.0, 0.0, 0.0
    eq = _equity_curve(initial_equity, trades)
    # Map trade pnl increments to bar returns approximately
    rets = eq.pct_change().dropna()
    if len(rets) < 2:
        return float(rets.sum()), 0.0, 0.0
    # Assume bars are daily
    ann_factor = 252
    ann_return = (1 + rets.mean())**ann_factor - 1
    ann_vol = float(rets.std() * (ann_factor**0.5))
    sharpe = float(0.0 if ann_vol == 0 else (rets.mean() * ann_factor) / ann_vol)
    return float(ann_return), float(ann_vol), sharpe


import pandas as pd
from .indicators import sma, atr

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma20"] = sma(out["close"], 20)
    out["sma50"] = sma(out["close"], 50)
    out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    # Signal: 1 if sma20 > sma50, else 0
    out["signal"] = (out["sma20"] > out["sma50"]).astype(int)
    return out


from dataclasses import dataclass

@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.005   # 0.5 percent of equity risked per trade
    atr_stop_mult: float = 2.0          # stop loss at 2 ATR below entry for long
    atr_take_mult: float = 3.0          # take profit at 3 ATR above entry for long
    daily_loss_cap_pct: float = 0.02    # stop trading for the day if loss exceeds 2 percent
    slippage_pct: float = 0.0005        # 5 bps slippage each way
    commission_per_trade: float = 0.0   # flat commission

def position_size_shares(equity: float, entry_price: float, atr_value: float, cfg: RiskConfig) -> int:
    # Dollar risk per share using ATR based stop
    dollar_risk_per_share = max(atr_value * cfg.atr_stop_mult, entry_price * 0.005)  # min tick risk floor
    risk_dollars = equity * cfg.risk_per_trade_pct
    shares = int(risk_dollars // dollar_risk_per_share)
    return max(shares, 0)


import pandas as pd

REQUIRED_COLS = ["time", "open", "high", "low", "close", "volume"]

def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Required: {REQUIRED_COLS}")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


import numpy as np

def simulate_gbm(S0: float, mu: float, sigma: float, steps: int = 252, dt: float = 1/252, paths: int = 1, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    S = np.zeros((steps + 1, paths))
    S[0] = S0
    for t in range(1, steps + 1):
        z = np.random.randn(paths)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return S


import pandas as pd
import numpy as np

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    # Wilder's smoothing approximation using EMA
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

from . import indicators, gbm, backtester, strategy, risk, broker_alpaca, utils
