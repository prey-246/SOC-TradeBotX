from src.backtester import Order, OrderBook
from typing import List
import pandas as pd

class Trader:
    def __init__(self):
        self.price_history = []
        self.max_position = 50
        self.quote_size = 15
        self.entry_price = None 
        self.position_side = None  # "long", "short", or None

    def run(self, state, current_position):
        orders: List[Order] = []
        order_depth: OrderBook = state.order_depth

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return {"PRODUCT": []}

        mid_price = (best_bid + best_ask) / 2
        self.price_history.append(mid_price)

        if len(self.price_history) < 35:
            return {"PRODUCT": []}

        prices = pd.Series(self.price_history)

        # Bollinger Bands
        sma = prices.rolling(20).mean().iloc[-1]
        std = prices.rolling(20).std().iloc[-1]
        upper = sma + 2 * std
        lower = sma - 2 * std

        # MACD
        ema_short = prices.ewm(span=12).mean()
        ema_long = prices.ewm(span=26).mean()
        macd_line = ema_short - ema_long
        macd_signal_line = macd_line.ewm(span=9).mean()

        macd_cross_up = macd_line.iloc[-2] < macd_signal_line.iloc[-2] and macd_line.iloc[-1] > macd_signal_line.iloc[-1]
        macd_cross_down = macd_line.iloc[-2] > macd_signal_line.iloc[-2] and macd_line.iloc[-1] < macd_signal_line.iloc[-1]

        # Z-score
        z = (mid_price - sma) / std if std != 0 else 0
        
        # Entry logic
        buy_signal = mid_price < lower and (z<-1.5 or macd_cross_up)
        sell_signal = mid_price > upper and (z>1.5 or macd_cross_down)

        buy_price = best_bid + 1.5
        sell_price = best_ask - 1.5

        # Open long
        if buy_signal and current_position + self.quote_size <= self.max_position and self.position_side is None:
            orders.append(Order("PRODUCT", buy_price, self.quote_size))
            self.entry_price = mid_price
            self.position_side = "long"

        # Open short
        if sell_signal and current_position - self.quote_size >= -self.max_position and self.position_side is None:
            orders.append(Order("PRODUCT", sell_price, -self.quote_size))
            self.entry_price = mid_price
            self.position_side = "short"

        # Exit conditions
        if self.position_side == "long" and mid_price >= sma:
            orders.append(Order("PRODUCT", sell_price, -self.quote_size))
            self.position_side = None
            self.entry_price = None

        if self.position_side == "short" and mid_price <= sma:
            orders.append(Order("PRODUCT", buy_price, self.quote_size))
            self.position_side = None
            self.entry_price = None

        return {"PRODUCT": orders}
