from typing import * 
from numbers import Number
import pandas as pd

class Indicator:
    def __init__(self, func, *args, overlay = False) -> None:
        self.overlay = overlay
        
        self.name = "Indicator " + getattr(func, '__name__', func.__class__.__name__) + f"({', '.join([str(arg) for arg in args if not isinstance(arg, (pd.Series))])})"
        
        try:
            self.values = func(*args)
        except Exception as e:
            raise RuntimeError(f"Error from {self.name}") from e

    def __repr__(self) -> str:
        return self.name

class Order:
    def __init__(self, num_shares, execute_price = None) -> None:
        self.num_shares = num_shares
        self.execute_price = execute_price
        # execute price is None means market order, assumed to be price at candle open

class Broker:
    def __init__(self, data) -> None:
        self._i = 0 
        self.data = data
        self.order_queue = []
        self.active_trades = []
        self.closed_trades = []
        
    def next(self):
        self._i += 1
        # Scan order queue and fill any orders based on new candlestick
        
        # Handle cash and equity changes, raise OutOfMoneyError if equity < 0
    
    def new_order(self, num_shares, limit_price = None, stop_loss = None, take_profit = None):
        # It is assumed that if it is a buy order with a SL / TP specified, buy first then sell at specified prices. 
        self.order_queue.append(Order(num_shares, limit_price))
        
        if stop_loss is not None:
            self.order_queue.append(Order(-num_shares, limit_price))
    
    def set_index(self, new_index):
        self._i = new_index

# Just a middleman between signals and broker
class Strategy:
    def __init__(self, data, broker) -> None:
        self._i = 0
        self.data = data
        self.broker = broker
        self.indicators = set()
    
    def next(self):
        self._i += 1
        
    def set_index(self, new_index):
        self._i = new_index

    def buy(self, num_shares, limit_price = None, stop_loss = None, take_profit = None):
        assert num_shares > 0, "num_shares must be > 0"
        return self.broker.new_order(num_shares, limit_price, stop_loss, take_profit)
    
    def sell(self, num_shares, limit_price = None, stop_loss = None, take_profit = None):
        assert num_shares > 0, "num_shares must be > 0"
        return self.broker.new_order(-num_shares, limit_price, stop_loss, take_profit)
    
    def crossover(self, series1: Sequence, series2: Sequence) -> bool:
        """
        Return `True` if `series1` just crossed over (above)
        `series2`.
        """
        def _adjust_length(series: Sequence):
            resize = False
            if isinstance(series, (pd.Series, Indicator)):
                series = series.values.tolist()
                if len(series) < 2:
                    resize = True
            if isinstance(series, Number):
                resize = True
            if resize:
                series = [series] * self._i
            return series

        series1 = _adjust_length(series1)
        series2 = _adjust_length(series2)
        try:
            return series1[self._i-1] < series2[self._i-1] and series1[self._i] > series2[self._i]
        except IndexError:
            return False
