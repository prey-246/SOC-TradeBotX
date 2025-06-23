from src.backtester import Order, OrderBook
import pandas as pd
from typing import List

class Trader:
    def __init__(self):
        self.price_history = []
        self.max_position = 50
        self.entry_price = None
        self.stop_loss = 100   
        self.take_profit = 100  

    def run(self, state, position):
        result = {}
        orders: List[Order] = []
        order_depth: OrderBook = state.order_depth

        # Get best bid and ask
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        # Early exit if no valid prices
        if best_bid is None or best_ask is None:
            result["PRODUCT"] = []
            return result

        mid_price = (best_bid + best_ask) / 2

        # Track price history
        if mid_price is not None:
            self.price_history.append(mid_price)

        prices = pd.Series(self.price_history)
        if len(prices) < 35:
            result["PRODUCT"] = []
            return result

        # --- Indicators ---
        # Moving averages
        short_ma = prices.rolling(window=10).mean().iloc[-1]
        long_ma = prices.rolling(window=20).mean().iloc[-1]

        # RSI
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14).mean().iloc[-1]
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12_series = prices.ewm(span=12, adjust=False).mean()
        ema26_series = prices.ewm(span=26, adjust=False).mean()
        macd_series = ema12_series - ema26_series
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        macd = macd_series.iloc[-1]
        signal = signal_series.iloc[-1]

        # Z-score 
        z_window = prices[-30:]
        if len(z_window) >= 30:
            z_mean = z_window.mean()
            z_std = z_window.std(ddof=0)
            z_score = (mid_price - z_mean) / z_std if z_std != 0 else 0
        else:
            z_score = 0

        # Bollinger Bands
        boll_ma = prices.rolling(window=20).mean().iloc[-1]
        boll_std = prices.rolling(window=20).std().iloc[-1]
        if pd.isna(boll_std) or boll_std == 0:
            upper_band = lower_band = mid_price
        else:
            upper_band = boll_ma + 2 * boll_std
            lower_band = boll_ma - 2 * boll_std

        # Entry Logic
        # Buy 
        val1 = int((short_ma > long_ma) + (macd > signal) + (z_score < -1.5) + (rsi < 35))
        if position < self.max_position and self.entry_price is None:
            if val1 >= 1 and mid_price <= (lower_band * 1.05):
                available_volume = abs(order_depth.sell_orders.get(best_ask, 0))
                volume = min(available_volume, self.max_position-position)
                if volume > 0:
                    orders.append(Order("PRODUCT", best_ask, volume))
                    self.entry_price = best_ask

        # Sell 
        val2 = int((short_ma < long_ma) + (macd < signal) + (z_score > 1.5) + (rsi > 65))
        if position > -self.max_position and self.entry_price is None:
            if val2 >= 1 and mid_price >= (upper_band * 0.95):
                available_volume = order_depth.buy_orders.get(best_bid, 0)
                volume = min(available_volume, self.max_position+position)
                if volume > 0:
                    orders.append(Order("PRODUCT", best_bid, -volume))
                    self.entry_price = best_bid

        # Exit
        if self.entry_price is not None:
            if position == 0:
                self.entry_price = None  
            else:
                pnl = (mid_price - self.entry_price)*position
                if pnl <= -self.stop_loss or pnl >= self.take_profit:
                    # Close the position
                    if position > 0 and best_bid:
                        close_volume = min(order_depth.buy_orders.get(best_bid, 0), position)
                        if close_volume > 0:
                            orders.append(Order("PRODUCT", best_bid, -close_volume))
                    elif position < 0 and best_ask:
                        close_volume = min(abs(order_depth.sell_orders.get(best_ask, 0)), abs(position))
                        if close_volume > 0:
                            orders.append(Order("PRODUCT", best_ask, close_volume))
                    self.entry_price = None

        result["PRODUCT"] = orders
        return result
