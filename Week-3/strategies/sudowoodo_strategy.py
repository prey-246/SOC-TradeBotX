from src.backtester import Order, OrderBook
from typing import List
import pandas as pd

class Trader:
    def __init__(self):
        self.price_history = []
        self.max_position = 50

    def run(self, state, current_position):
        orders: List[Order] = []
        order_depth: OrderBook = state.order_depth

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return {"PRODUCT": []}

        mid_price = (best_bid + best_ask) / 2
        self.price_history.append(mid_price)

        if len(self.price_history) < 20:
            return {"PRODUCT": []}

        prices = pd.Series(self.price_history)
        mean = prices.rolling(20).mean().iloc[-1]
        std = prices.rolling(20).std().iloc[-1]

        z = (mid_price - mean) / std if std != 0 else 0
        
        upper = mean + 2 * std
        lower = mean - 2 * std

        sma10 = prices.rolling(window=10).mean()
        sma20 = prices.rolling(window=20).mean()

        prev_sma10 = sma10.iloc[-2]
        prev_sma20 = sma20.iloc[-2]
        curr_sma10 = sma10.iloc[-1]
        curr_sma20 = sma20.iloc[-1]

        crossover_buy = prev_sma10 < prev_sma20 and curr_sma10 > curr_sma20
        crossover_sell = prev_sma10 > prev_sma20 and curr_sma10 < curr_sma20

        quote_size = 15

        # Buy
        if (z < -2 or mid_price < lower or crossover_buy or current_position + quote_size <= self.max_position):
            orders.append(Order("PRODUCT", 9998, quote_size))

        # Sell
        if (z > 2 or mid_price > upper or crossover_sell or current_position - quote_size >= -self.max_position):
            orders.append(Order("PRODUCT", 10002, -quote_size))

        return {"PRODUCT": orders}
