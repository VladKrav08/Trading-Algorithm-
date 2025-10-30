
import argparse, time, os
import pandas as pd
from datetime import datetime
from algo.broker_alpaca import AlpacaConfig, get_client, place_bracket_order
from algo.risk import RiskConfig, position_size_shares
from algo.indicators import sma, atr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--risk_perc", type=float, default=0.005)
    p.add_argument("--stop_mult", type=float, default=2.0)
    p.add_argument("--take_mult", type=float, default=3.0)
    p.add_argument("--equity", type=float, default=100000.0)
    args = p.parse_args()

    cfg = AlpacaConfig(symbol=args.symbol)
    client = get_client(cfg)

    # Pull last 100 daily bars
    bars = client.get_bars(args.symbol, "1Day", limit=100).df
    bars = bars.reset_index()
    bars.rename(columns={"timestamp":"time","open":"open","high":"high","low":"low","close":"close","volume":"volume"}, inplace=True)
    bars["time"] = pd.to_datetime(bars["time"])

    bars["sma20"] = sma(bars["close"], 20)
    bars["sma50"] = sma(bars["close"], 50)
    bars["atr14"] = atr(bars["high"], bars["low"], bars["close"], 14)
    last = bars.dropna().iloc[-1]

    if last["sma20"] <= last["sma50"]:
        print("No long signal. Exiting.")
        return

    price = float(last["close"])
    atr_val = float(last["atr14"])
    r = RiskConfig(risk_per_trade_pct=args.risk_perc, atr_stop_mult=args.stop_mult, atr_take_mult=args.take_mult)
    shares = position_size_shares(args.equity, price, atr_val, r)
    if shares <= 0:
        print("Shares computed to 0. Exiting.")
        return

    stop_price = price - r.atr_stop_mult * atr_val
    take_price = price + r.atr_take_mult * atr_val
    order = place_bracket_order(client, args.symbol, shares, "buy", price, stop_price, take_price)
    print("Submitted order id:", getattr(order, "id", order))

if __name__ == "__main__":
    main()
