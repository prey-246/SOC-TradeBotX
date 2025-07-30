from AlgoTradingBacktester.src.backtester import Order, OrderBook
from typing import List
import pandas as pd
import numpy as np
import statistics

# Base Class
class BaseClass:
    def __init__(self, product_name, max_position):
        self.product_name = product_name
        self.max_position = max_position
    
    def get_orders(self, state, orderbook, position):
        """Override this method in product-specific strategies"""
        return []

class AbraStrategy(BaseClass):
    def __init__(self):
        super().__init__("ABRA", 50)
        self.prices = []
        self.lookback = 200
        self.z_threshold = 2.0
        self.z_mm_threshold = 0.3
        self.skew_factor = 0.1
    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if len(self.prices) > self.lookback:
            mean_price = statistics.mean(self.prices[-self.lookback:])
            stddev_price = statistics.stdev(self.prices[-self.lookback:])
            z_score = (mid_price - mean_price) / stddev_price
            if z_score > self.z_threshold:
                orders.append(Order(self.product_name, best_bid, -7))
            elif z_score < -self.z_threshold:
                orders.append(Order(self.product_name, best_ask, 7))
            elif abs(z_score) < self.z_mm_threshold:
                return self.market_make(mid_price, position)
        elif len(self.prices) <= self.lookback:
            return self.market_make(mid_price, position)
        return orders

    def market_make(self, mid_price, position):
        orders = []
        adjusted_mid_price = mid_price + self.skew_factor*position
        orders.append(Order(self.product_name, adjusted_mid_price - 2, 7))
        orders.append(Order(self.product_name, adjusted_mid_price + 2, -7))
        return orders

class AshStrategy(BaseClass):
    def __init__(self):
        super().__init__("ASH", 60)
        self.value_size = 1
        self.min_spread = 8
        self.entry_price = None
        self.stop_loss_pct = 0.002 
        self.max_loss_per_trade=265 

    def get_orders(self, state, orderbook, position):
        orders = []


        if not orderbook.buy_orders or not orderbook.sell_orders:
            return orders

        best_bid = max(orderbook.buy_orders.keys())
        best_ask = min(orderbook.sell_orders.keys())
        bid_vol = orderbook.buy_orders[best_bid]
        ask_vol = orderbook.sell_orders[best_ask]
        mid_price = (best_bid + best_ask) / 2

        if position == 0:
            self.entry_price = None
        elif self.entry_price is None:
            self.entry_price = mid_price

        if self.entry_price is not None and position != 0:
            unrealized_loss = abs(mid_price - self.entry_price) * abs(position)
            lossing= self.max_loss_per_trade * abs(position) 
            if unrealized_loss >= lossing:                
                if position > 0:
                    orders.append(Order(self.product_name, best_bid, -abs(position))) 
                else:
                    orders.append(Order(self.product_name, best_ask, abs(position)))   

                self.entry_price = None
                return orders

        if best_ask - best_bid >= self.min_spread:
            if position < self.max_position:
                orders.append(Order(self.product_name, best_bid, min(self.value_size, bid_vol)))
                if position == 0 and self.entry_price is None:
                    self.entry_price = best_bid  
            if position > -self.max_position:
                orders.append(Order(self.product_name, best_ask, -min(self.value_size, ask_vol)))
                if position == 0 and self.entry_price is None:
                    self.entry_price = best_ask  

        return orders

class DrowzeeStrategy(BaseClass):
    def __init__(self):
        super().__init__("DROWZEE", 50)
        self.lookback = 200
        self.z_threshold = 3.75
        self.prices = []

    def get_orders(self, state, orderbook, position):
        orders = []
        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if len(self.prices) > self.lookback:
            mean_price = statistics.mean(self.prices[-self.lookback:])
            stddev_price = statistics.stdev(self.prices[-self.lookback:])
            z_score = (mid_price - mean_price) / stddev_price
            if z_score > self.z_threshold:
                orders.append(Order(self.product_name, best_bid, -self.max_position + position))
            elif z_score < -self.z_threshold:
                orders.append(Order(self.product_name, best_ask, self.max_position - position))
            else:
                return self.market_make(mid_price)
        elif len(self.prices) <= self.lookback:
            return self.market_make(mid_price)
        return orders

    def market_make(self, mid_price):
        orders = []
        orders.append(Order(self.product_name, mid_price - 1, 25))
        orders.append(Order(self.product_name, mid_price + 1, -25))
        return orders

class JolteonStrategy(BaseClass):
    def __init__(self):
        super().__init__("JOLTEON", 350)
        self.prices = []
        self.lookback = 120
        self.value_size = 5
        self.skew_factor = 0.2
        self.z_threshold = 1.8
        self.rsi_low = 35
        self.rsi_high = 65

    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders or not orderbook.sell_orders:
            return orders

        best_bid = max(orderbook.buy_orders)
        best_ask = min(orderbook.sell_orders)
        mid_price = (best_bid + best_ask) // 2
        spread = best_ask - best_bid
        self.prices.append(mid_price)

        if len(self.prices) <= self.lookback:
            return self.market_make(mid_price, position, spread)

        prices = pd.Series(self.prices[-self.lookback:])
        sma = prices.mean()
        std = prices.std() or 1
        z = (mid_price - sma) / std

        # RSI
        delta = prices.diff().dropna()
        up = delta.where(delta > 0, 0.0)
        down = -delta.where(delta < 0, 0.0)
        avg_gain = up.rolling(14).mean().iloc[-1]
        avg_loss = down.rolling(14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50

        # Entry/Exit signals
        buy_signal = z < -self.z_threshold and rsi < self.rsi_low and position < self.max_position
        sell_signal = z > self.z_threshold and rsi > self.rsi_high and position > -self.max_position

        if buy_signal:
            qty = min(self.value_size, self.max_position - position)
            orders.append(Order(self.product_name, best_ask, qty))

        elif sell_signal:
            qty = min(self.value_size, self.max_position + position)
            orders.append(Order(self.product_name, best_bid, -qty))

        elif abs(z) < 0.25 and 45 < rsi < 55 and abs(position) < self.max_position * 0.8:
            return self.market_make(mid_price, position, spread)

        return orders

    def market_make(self, mid_price, position, spread):
        orders = []

        if spread < 2:
            return orders

        skew = self.skew_factor * (position / self.max_position)
        fair_bid = round(mid_price + skew - 1)
        fair_ask = round(mid_price + skew + 1)

        if position < self.max_position:
            orders.append(Order(self.product_name, fair_bid, self.value_size))
        if position > -self.max_position:
            orders.append(Order(self.product_name, fair_ask, -self.value_size))

        return orders

class LuxrayStrategy(BaseClass):
    def __init__(self):
        super().__init__("LUXRAY", 250)
        self.prices = []
        self.lookback = 50
        self.skew_factor = 0.1
        self.value_size = 100
        self.entry_price = None  

    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        spread = best_ask - best_bid
        self.prices.append(mid_price)


  
        if len(self.prices) > self.lookback:
            prices = pd.Series(self.prices[-self.lookback:])

            sma = prices.rolling(self.lookback).mean().iloc[-1]
            std = prices.rolling(self.lookback).std().iloc[-1]
            sma10 = prices.rolling(10).mean().iloc[-1]
            sma20 = prices.rolling(20).mean().iloc[-1]
            sma_trend_up = sma10 > sma20
            sma_trend_down = sma10 < sma20
            # MACD
            ema_short = prices.ewm(span=12).mean()
            ema_long = prices.ewm(span=26).mean()
            macd_line = ema_short - ema_long
            macd_signal_line = macd_line.ewm(span=9).mean()
            macd_cross_up = (
                macd_line.iloc[-2] < macd_signal_line.iloc[-2]
                and macd_line.iloc[-1] > macd_signal_line.iloc[-1]
            )
            macd_cross_down = (
                macd_line.iloc[-2] > macd_signal_line.iloc[-2]
                and macd_line.iloc[-1] < macd_signal_line.iloc[-1]
            )

            # Z-score
            z_score = (mid_price - sma) / std if std != 0 else 0

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

            # Combined Signal Logic
            buy_signal = (
                rsi < 37 or
                macd_cross_up or
                z_score < -2 or sma_trend_up
            )
            sell_signal = (
                rsi > 67
                or macd_cross_down or z_score>2 or sma_trend_down
            )
            no_signal = (
                45 < rsi < 55 or abs(z_score) < 0.4)
            
            buy_price = mid_price - (spread *0.8)/ 2
            sell_price = mid_price + (spread*0.8) / 2
            if buy_signal or position + self.value_size <= self.max_position:
                orders.append(Order(self.product_name, int(buy_price), self.max_position - position))
            elif sell_signal or position - self.value_size >= -self.max_position:
                orders.append(Order(self.product_name, int(sell_price), -self.max_position + position))
            elif no_signal:
                return self.market_make(spread, mid_price, position)
            
        else:
            return self.market_make(spread, mid_price, position)
        return orders

    def market_make(self,spread, mid_price, position):
        orders = []
        adjusted_mid_price = mid_price + self.skew_factor * position
        buy_price = adjusted_mid_price - (spread ) / 2
        sell_price = adjusted_mid_price + (spread) / 2
        orders.append(Order(self.product_name, int(buy_price), self.value_size))
        orders.append(Order(self.product_name, int(sell_price), -self.value_size))
        return orders
    
class MistyStrategy(BaseClass):
    def __init__(self):
        super().__init__("MISTY", 100)
        self.prices = []
        self.lookback = 300
        self.z_mm_threshold = 0.3
        self.skew_factor = 0.5
        self.value_size = 1
        self.entry_price = None
        self.per_unit_tp = 8


    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders or not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys())
        best_bid = max(orderbook.buy_orders.keys())
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if position == 0:
            self.entry_price = None
        elif self.entry_price is None:
            self.entry_price = mid_price

        # Exit
        if self.entry_price is not None and position != 0:
            direction = 1 if position > 0 else -1
            exit_price = best_bid if direction == 1 else best_ask
            pnl_per_unit = (exit_price - self.entry_price) * direction

            if pnl_per_unit >= self.per_unit_tp:
                orders.append(Order(self.product_name, exit_price, -position))
                self.entry_price = None
                return orders

        if len(self.prices) > self.lookback:
            prices = pd.Series(self.prices[-self.lookback:])

            # Bollinger Bands
            sma = prices.rolling(self.lookback).mean().iloc[-1]
            std = prices.rolling(self.lookback).std().iloc[-1]
            upper = sma + 2 * std
            lower = sma - 2 * std

            # MACD
            ema_short = prices.ewm(span=12).mean()
            ema_long = prices.ewm(span=26).mean()
            macd_line = ema_short - ema_long
            macd_signal = macd_line.ewm(span=9).mean()
            macd_cross_up = macd_line.iloc[-2] < macd_signal.iloc[-2] and macd_line.iloc[-1] > macd_signal.iloc[-1]
            macd_cross_down = macd_line.iloc[-2] > macd_signal.iloc[-2] and macd_line.iloc[-1] < macd_signal.iloc[-1]

            # Z-score
            z_score = (mid_price - sma) / std if std != 0 else 0

            # RSI
            delta = prices.diff().dropna()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]
            if avg_loss == 0 or np.isnan(avg_loss):
                rsi = 100 if avg_gain > 0 else 50
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Signal logic
            buy_signal = sum([
                macd_cross_up,
                mid_price < lower,
                rsi < 35,
                z_score < -2.3
            ]) >= 2

            sell_signal = sum([
                macd_cross_down,
                mid_price > upper,
                rsi > 65,
                z_score > 2.3
            ]) >= 2

            if buy_signal:
                orders.append(Order(self.product_name, best_ask, self.value_size))
                if self.entry_price is None:
                    self.entry_price = best_ask
            elif sell_signal:
                orders.append(Order(self.product_name, best_bid, -self.value_size))
                if self.entry_price is None:
                    self.entry_price = best_bid
            elif 45 < rsi < 55 and abs(z_score) < self.z_mm_threshold:
                return self.market_make(mid_price, position)
        else:
            return self.market_make(mid_price, position)

        return orders

    def market_make(self, mid_price, position):
        orders = []
        skewed_mid = mid_price + self.skew_factor * position
        orders.append(Order(self.product_name, skewed_mid - 1, self.value_size))
        orders.append(Order(self.product_name, skewed_mid + 1, -self.value_size))
        return orders

class ShinxStrategy(BaseClass):
    def __init__(self):
        super().__init__("SHINX", 50)
        self.prices = []
        self.lookback = 200
        self.z_threshold = 2.0
        self.z_mm_threshold = 0.3
        self.skew_factor = 0.1
        self.value_size = 5

    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if len(self.prices) > self.lookback:
            prices = pd.Series(self.prices[-self.lookback:])
            sma = prices.rolling(self.lookback).mean().iloc[-1]
            std = prices.rolling(self.lookback).std().iloc[-1]

            # MACD
            ema_short = prices.ewm(span=12).mean()
            ema_long = prices.ewm(span=26).mean()
            macd_line = ema_short - ema_long
            macd_signal_line = macd_line.ewm(span=9).mean()
            macd_cross_up = (
                macd_line.iloc[-2] < macd_signal_line.iloc[-2]
                and macd_line.iloc[-1] > macd_signal_line.iloc[-1]
            )
            macd_cross_down = (
                macd_line.iloc[-2] > macd_signal_line.iloc[-2]
                and macd_line.iloc[-1] < macd_signal_line.iloc[-1]
            )

            # Z-score
            z_score = (mid_price - sma) / std if std != 0 else 0

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

            
            buy_signal = (
                (macd_cross_up)
                and (rsi < 35 or z_score < -1.5) 
            )
            sell_signal = (
                (macd_cross_down)
                and (rsi > 65 or z_score > 1.5)  

            )
          
            if buy_signal:
                orders.append(Order(self.product_name, best_ask, self.value_size))
            elif sell_signal:
                orders.append(Order(self.product_name, best_bid, -self.value_size))
            elif 45 < rsi < 55 and abs(z_score) < self.z_mm_threshold:
                return self.market_make(mid_price, position)
        else:
            return self.market_make(mid_price, position)
        return orders

    def market_make(self, mid_price, position):
        orders = []
        adjusted_mid_price = mid_price + self.skew_factor * position
        orders.append(Order(self.product_name, adjusted_mid_price - 1 , self.value_size))
        orders.append(Order(self.product_name, adjusted_mid_price + 1, -self.value_size))
        return orders
    
class SudowoodoStrategy(BaseClass): 

    def __init__(self):
        super().__init__("SUDOWOODO", 50)
        self.fair_value = 10000
    
    def get_orders(self, state, orderbook, position):
        orders = []
        
        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        orders.append(Order(self.product_name, self.fair_value + 2, -10))
        orders.append(Order(self.product_name, self.fair_value - 2, 10))

        return orders
    
class Trader:
    MAX_LIMIT = 0 # for single product mode only, don't remove
    def __init__(self):
        self.strategies = {
            "ABRA": AbraStrategy(),
            "ASH": AshStrategy(),
            "DROWZEE": DrowzeeStrategy(),
            "JOLTEON": JolteonStrategy(),
            "LUXRAY": LuxrayStrategy(),
            "MISTY": MistyStrategy(),
            "SHINX": ShinxStrategy(),
            "SUDOWOODO": SudowoodoStrategy()
        }
    
    def run(self, state):
        result = {}
        positions = getattr(state, 'positions', {})
        if len(self.strategies) == 1: self.MAX_LIMIT= self.strategies["PRODUCT"].max_position # for single product mode only, don't remove

        for product, orderbook in state.order_depth.items():
            current_position = positions.get(product, 0)
            product_orders = self.strategies[product].get_orders(state, orderbook, current_position)
            result[product] = product_orders
        
        return result, self.MAX_LIMIT
