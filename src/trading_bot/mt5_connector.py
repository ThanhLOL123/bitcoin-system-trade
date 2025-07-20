from mt5linux import MetaTrader5
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

class MT5Connector:
    """Connector for MetaTrader 5 using mt5linux"""
    
    def __init__(self, host, port, login, password, server):
        self.mt5 = MetaTrader5(host=host, port=port)
        self.login = login
        self.password = password
        self.server = server

    def connect(self):
        """Connect to MetaTrader 5"""
        if not self.mt5.initialize(login=self.login, password=self.password, server=self.server):
            print("initialize() failed, error code =", self.mt5.last_error())
            return False
        
        return True

    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        self.mt5.shutdown()

    def place_order(self, symbol: str, order_type: str, volume: float, price: float, sl: float = 0.0, tp: float = 0.0):
        """Place a market order"""
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 234000,
            "comment": "python script",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        if order_type == "BUY":
            request["type"] = self.mt5.ORDER_TYPE_BUY
        elif order_type == "SELL":
            request["type"] = self.mt5.ORDER_TYPE_SELL
        
        result = self.mt5.order_send(request)
        return result

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = self.mt5.positions_get()
        if positions is None:
            return []
        return [pos._asdict() for pos in positions]

    def get_historical_data(self, symbol: str, timeframe: Any, bars: int) -> pd.DataFrame:
        """Get historical data"""
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            return pd.DataFrame()
        return pd.DataFrame(rates)

    def account_info(self):
        return self.mt5.account_info()

    def symbol_info(self, symbol):
        return self.mt5.symbol_info(symbol)

    def symbol_info_tick(self, symbol):
        return self.mt5.symbol_info_tick(symbol)