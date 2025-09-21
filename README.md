# Trading-Algorithm-
📈 Trading Algorithm

This is an algorithmic trading system I created using Python.
It demonstrates how to build, backtest, and run a trading strategy with live paper trading through Alpaca Markets.


🚀 Features
	•	Moving Average Crossover strategy (MA 20 vs MA 50)
	•	ATR-based stop-loss and take-profit logic
	•	Volatility-adjusted position sizing (risk per trade = % of equity)
	•	Daily loss cap to prevent large drawdowns
	•	Backtesting mode (using CSV OHLCV data)
	•	Live paper trading mode (via Alpaca API)

🛠️ Setup
	1.	Clone this repository
	2.	Install dependencies:
 pip install pandas numpy matplotlib alpaca-trade-api python-dotenv pytz
  3.	Create a .env file in the project root (do not commit real keys):
  ALPACA_API_KEY_ID=YOUR_KEY
  ALPACA_API_SECRET_KEY=YOUR_SECRET
  ALPACA_PAPER_BASE=https://paper-api.alpaca.markets
  4.	Prepare a CSV (for backtesting) with columns:
  time,open,high,low,close,volume
  Backtest
  python algo.py --mode BACKTEST --csv data.csv
  Live Paper Trading
  python algo.py --mode LIVE_PAPER
  📊 Example Metrics from Backtest
	•	CAGR
	•	Sharpe Ratio
	•	Max Drawdown
	•	Final Equity
