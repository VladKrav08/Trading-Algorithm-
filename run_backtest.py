
import argparse
from algo.utils import load_price_csv
from algo.backtester import backtest, BacktestConfig
from algo.risk import RiskConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV with time, open, high, low, close, volume")
    p.add_argument("--init_equity", type=float, default=100000.0)
    p.add_argument("--risk_perc", type=float, default=0.005)
    p.add_argument("--stop_mult", type=float, default=2.0)
    p.add_argument("--take_mult", type=float, default=3.0)
    p.add_argument("--daily_cap", type=float, default=0.02)
    args = p.parse_args()

    df = load_price_csv(args.csv)
    cfg = BacktestConfig(
        initial_equity=args.init_equity,
        risk=RiskConfig(
            risk_per_trade_pct=args.risk_perc,
            atr_stop_mult=args.stop_mult,
            atr_take_mult=args.take_mult,
            daily_loss_cap_pct=args.daily_cap
        )
    )
    result = backtest(df, cfg)

    print("Ending equity:", round(result["ending_equity"], 2))
    print("Total return:", round(result["total_return"] * 100, 2), "%")
    print("Max drawdown:", round(result["max_drawdown"] * 100, 2), "%")
    print("Annualized return:", round(result["ann_return"] * 100, 2), "%")
    print("Annualized volatility:", round(result["ann_vol"] * 100, 2), "%")
    print("Sharpe like:", round(result["sharpe_like"], 2))
    print("Trades:", len(result["trades"]))

if __name__ == "__main__":
    main()
