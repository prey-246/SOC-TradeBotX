from src.backtester import Order, OrderBook
from typing import List
import pandas as pd
import numpy as np

class Trader:
    def __init__(self):
        self.max_position = 50
        self.quote_size = 37 
        self.min_spread_threshold = 1 
        self.price_history = []

    def run(self, state, current_position):
        orders: List[Order] = []
        order_depth: OrderBook = state.order_depth

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return {"PRODUCT": []}

        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        self.price_history.append(mid_price)

        if len(self.price_history) < 30:
            return {"PRODUCT": []}

        prices = pd.Series(self.price_history)

        # MACD
        ema_short = prices.ewm(span=12).mean()
        ema_long = prices.ewm(span=26).mean()
        macd_line = ema_short - ema_long
        macd_signal_line = macd_line.ewm(span=9).mean()

        macd_cross_up = macd_line.iloc[-2] < macd_signal_line.iloc[-2] and macd_line.iloc[-1] > macd_signal_line.iloc[-1]
        macd_cross_down = macd_line.iloc[-2] > macd_signal_line.iloc[-2] and macd_line.iloc[-1] < macd_signal_line.iloc[-1]

        # Z-score
        sma = prices.rolling(20).mean().iloc[-1]
        std = prices.rolling(20).std().iloc[-1]
        z = (mid_price - sma) / std if std != 0 else 0

        # RSI 
        delta = prices.diff().dropna()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        avg_gain = gains.rolling(14).mean().iloc[-1]
        avg_loss = losses.rolling(14).mean().iloc[-1]
            
        if avg_loss == 0 or np.isnan(avg_loss):
            rsi = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs)) if not np.isinf(rs) else 100

        if spread >= self.min_spread_threshold:
            buy_price = mid_price - spread / 2
            sell_price = mid_price + spread / 2

            buy_signal = z<-1.5 or macd_cross_up or rsi < 35
            sell_signal = z>1.5 or macd_cross_down or rsi > 65

            # Buy & Sell
            if buy_signal or current_position + self.quote_size <= self.max_position:
                orders.append(Order("PRODUCT", int(buy_price), self.quote_size))

            if sell_signal or current_position - self.quote_size >= -self.max_position:
                orders.append(Order("PRODUCT", int(sell_price), -self.quote_size))

        return {"PRODUCT": orders}
