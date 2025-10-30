
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
