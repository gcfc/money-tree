from typing import * 
from numbers import Number
import pandas as pd
import numpy as np
from math import copysign

class OutOfMoneyError(Exception):
    pass

class Indicator:
    def __init__(self, func, *args, overlay = False) -> None:
        self.overlay = overlay
        self.name = getattr(func, '__name__', func.__class__.__name__) + f"({', '.join([str(arg) for arg in args if not isinstance(arg, (pd.Series))])})"
        
        try:
            self.values = func(*args)
        except Exception as e:
            raise RuntimeError(f"Error from {self.name}") from e

    def __repr__(self) -> str:
        return self.name

# Receives commands from Broker, modify Broker's order queue

# If SL / TP are specified in orders, Trade class is the one that coordinates
# the corresponding order to close.
class Trade:
    def __init__(self, broker, num_shares, entry_price, entry_index, comment: str = None) -> None:
        self.broker = broker
        self.num_shares = num_shares
        self.entry_prices = [entry_price]
        self.entry_indices = [entry_index]
        self.entry_shares_hist = [num_shares]
        
        self.exit_prices = []
        self.exit_indices = []
        self.exit_shares_hist = []
        self.comment = comment
        self.stop_loss_order = None
        self.take_profit_order = None
        self.avg_cost = entry_price
        self.pnl = 0
        self.pnl_history = []

    def __repr__(self):
        return f'<Trade size={self.num_shares} time={self.broker.data.Datetime[self.entry_indices[0]]}-{self.broker.data.Datetime[self.exit_indices[-1]] if self.exit_indices else ""} ' \
               f'avg_cost={self.avg_cost:.2f}-{self.exit_prices[-1] if self.exit_prices else ""} pnl={self.pnl:.2f}' \
               f'{" comment="+(self.comment if self.comment is not None else "")}>'
    
    def update_pnl(self):
        # TODO : Unit test this.
        in_trade = self.num_shares * self.broker.last_price()
        sell_price = sum(price * shares for price, shares in zip(self.exit_prices, self.exit_shares_hist))
        buy_price = sum(price * shares for price, shares in zip(self.entry_prices, self.entry_shares_hist)) 
        sgn = copysign(1, self.num_shares)
        self.pnl = sgn * (in_trade + sell_price - buy_price)
        return self.pnl
    
    def pnl_percentage(self):
        # TODO: fix this math
        price = self.broker.last_price() # or exit price
        sgn = copysign(1, self.num_shares)
        return sgn * (price - self.avg_cost) / self.avg_cost

# So far only support market order and limit orders. Stop loss orders are simulated as limit orders at specified price. 
class Order:
    def __init__(self, num_shares, limit_price = None, stop_price = None, stop_loss = None, take_profit = None, parent_trade: Trade = None, comment: str = None) -> None:
        # Positive num_shares means buy order, negative means sell
        self.num_shares = num_shares
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        # if no price is set, that means market order, assumed to be price at candle open 
        # (or previous candle close if trade_on_close set to True)
        self.parent_trade = parent_trade
        self.comment = comment

# TODO: to be implemented
class StopLoss:
    def __init__(self, price, trail_percentage=None, trail_amount=None) -> None:
        self.price = price
        self.trail_percentage = trail_percentage # Range: [0, 100]
        self.trail_amount = trail_amount # Dollar amount
        assert not (self.trail_percentage and self.trail_amount), "Specify only one of trail_percentage or trail_amount!"

    
# A middleman between strategy output (buy / sell signal at specific prices) and Trades
class Broker:
    def __init__(self, data, trade_on_close=False, init_cash=2000) -> None:
        self._i = 0 
        self.data = data
        self.order_queue: List[Order] = []
        self.trades: Dict[Union[str, None], Trade] = dict()
        self.closed_trades: List[Trade] = []
        # trade_on_close = True means market orders are executed on the previous candle close. 
        # False means market orders are executed on the current candle's open. Defaults to False. 
        self.trade_on_close = trade_on_close
        self.cash = init_cash
        self.equity = 0
        self.equity_history = []
        
    def next(self, verbose=False):
        # Scan order queue and fill any orders based on new candlestick, results in Trades added to self.trades
        candle = self.data.iloc[self._i]
        curr_open, curr_high, curr_low, curr_close = candle.Open, candle.High, candle.Low, candle.Close
        prev_close = self.data.iloc[self._i-1].Close
        for order in list(self.order_queue): 
            # Copy the initial order queue and iterate on the version before any deletions.  
            if order not in self.order_queue:
                continue

            execute_order = False
            execute_price = None
            time_index = self._i
            
            # Is it a market order? 
            if order.limit_price is None and order.stop_price is None and order.take_profit is None:
                execute_order = True
                execute_price = prev_close if self.trade_on_close else curr_open
                time_index = self._i-1 if self.trade_on_close else self._i
            
            # Is it a limit order? 
            elif order.limit_price is not None and \
              ((order.num_shares > 0 and curr_low <= order.limit_price) \
              or (order.num_shares < 0 and curr_high >= order.limit_price)): 
                # Long orders and short orders have different criteria
                execute_order = True
                execute_price = order.limit_price
            
            # Is it a stop order?
            elif order.stop_price is not None and \
              ((order.num_shares > 0 and curr_high >= order.stop_price) \
              or (order.num_shares < 0 and curr_low <= order.stop_price)): 
                # Long orders and short orders have different criteria
                execute_order = True
                execute_price = order.stop_price
            
            if not execute_order:
                continue
            
            # Actually execute order
            assert execute_price is not None, "Execute price not set!"
            
            # If order is a SL/TP order, add to / reduce / close the trades when applicable
            if order.parent_trade is not None:
                self.adjust_current_trade(order.parent_trade, order.num_shares, execute_price, time_index, order.comment)
            
            # Be intentional with the comments. Orders with the same comments are grouped together into one trade. Orders without comments grouped into its own bucket. 
            # TODO: Unit test this. 
            else:
                if order.comment not in self.trades:
                    # Open a new trade
                    self.open_trade(order.num_shares, execute_price, time_index, order.comment, order.stop_loss, order.take_profit)
                    
                else:
                    # Update the corresponding trade based on comments. Note that the associated trades by same comment are different from "parent trades".
                    
                    # original_size = self.trades[order.comment].num_shares
                    # new_size = max(-original_size, order.num_shares) if original_size > 0 else min(-original_size, order.num_shares)
                    curr_trade = self.trades[order.comment]
                    resultant_size = curr_trade.num_shares + order.num_shares
                    if resultant_size * curr_trade.num_shares >= 0: # Does not reverse direction
                        self.adjust_current_trade(curr_trade, order.num_shares, execute_price, time_index, order.comment)
                    else: 
                        # Close out the current trade and open a new one in the other direction
                        new_trade_size = order.num_shares - (-curr_trade.num_shares)
                        assert new_trade_size * curr_trade.num_shares < 0, "Direction not reversed, implementation error"
                        self.adjust_current_trade(curr_trade, -curr_trade.num_shares, execute_price, time_index, order.comment)
                        self.open_trade(new_trade_size, execute_price, time_index, order.comment, order.stop_loss, order.take_profit)
                    
            self.order_queue.remove(order)

        # Handle cash and equity changes, raise OutOfMoneyError if equity < 0
        for _, trade in self.trades.items():
            trade.update_pnl()
            trade.pnl_history.append(trade.pnl)
        self.equity = self.cash + sum(trade.pnl for trade in self.trades.values())
        self.equity_history.append(self.equity)
        if self.equity <= 0:
            for _, trade in dict(self.trades).items():
                self.close_trade(trade, curr_close, self._i)
            assert len(self.trades) == 0
            self.cash = 0
            return OutOfMoneyError("Out of money! Stopping simulation...")
        
        if verbose:
            print("Time:\t", self.data.Datetime[self._i], "Last price:", round(self.last_price(), 2))
            print("Open trades:\t", self.trades)
            print("Close trades:\t", self.closed_trades)
            print()

        self._i += 1
        
    def new_order(self, num_shares, limit_price = None, stop_price = None, stop_loss = None, take_profit = None, parent_trade = None, comment = None):
        # Only support whole number shares at the moment
        assert num_shares == round(num_shares), f"Share size {num_shares} must be whole number."
        num_shares = int(num_shares)
        
        # Validation of input values
        for arg in {limit_price, stop_price, stop_loss, take_profit}:
            assert arg is None or arg > 0, f"Order value error: {arg}"
        
        prev_close = self.data.iloc[self._i-1].Close
        curr_open = self.data.iloc[self._i].Open
        potential_execute_price = prev_close if self.trade_on_close else curr_open
        if num_shares > 0:
            if not (stop_loss or -np.inf) < (limit_price or stop_price or potential_execute_price) < (take_profit or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({stop_loss}) < PRICE ({limit_price or stop_price or potential_execute_price}) < TP ({take_profit})")
        else:
            if not (take_profit or -np.inf) < (limit_price or stop_price or potential_execute_price) < (stop_loss or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({take_profit}) < PRICE ({limit_price or stop_price or potential_execute_price}) < SL ({stop_loss})")

        # This function appends an order to the order_queue with all info in one Order object. 
        # .next() function splits this into multiple orders if applicable, with SL/TP firing later. 
        order = Order(num_shares, limit_price, stop_price, stop_loss, take_profit, parent_trade, comment)
        
        # Only broken down SL/TP orders have parent trades specified. Put them at the front of the queue. 
        if parent_trade is not None:
            self.order_queue.insert(0, order)
        else:
            self.order_queue.append(order)
        return order
          
    def set_index(self, new_index):
        self._i = new_index
    
    def adjust_current_trade(self, trade: Trade, new_shares: int, price: float, time_index: int, comment: Union[None, str]):
        # adjust trade size and PNL
        # This function cannot reverse the direction of the original trade. 
        # Must explicitly place another order.         
        resultant_size = trade.num_shares + new_shares
        assert resultant_size * trade.num_shares > 0, "Illegal operation, trade direction reversed!"
        if resultant_size == 0: 
            self.close_trade(trade, price, time_index)
        else:
            if trade.num_shares * new_shares > 0: # Adding to the original direction of trade
                trade.entry_prices.append(price)
                trade.entry_indices.append(time_index)
                trade.entry_shares_hist.append(new_shares)
                trade.avg_cost = (trade.avg_cost * trade.num_shares + price * new_shares) / resultant_size
            else:
                trade.exit_prices.append(price)
                trade.exit_indices.append(time_index)
                trade.exit_shares_hist.append(new_shares)
            trade.num_shares += new_shares
            # If originally a long trade, adjust the cash
            # If originally a short trade, "borrowed shares" are not cash, and covering does not change the cash, for simplicity. 
            if trade.num_shares > 0:
                self.cash -= price * new_shares 
    
    def close_trade(self, trade: Trade, price: float, time_index: int):
        trade = self.trades.pop(trade.comment)
        trade.exit_prices.append(price)
        trade.exit_indices.append(time_index)
        trade.exit_shares_hist.append(trade.num_shares)
        if trade.stop_loss_order:
            self.orders.remove(trade.stop_loss_order)
        if trade.take_profit_order:
            self.orders.remove(trade.take_profit_order)

        self.closed_trades.append(trade)
        self.cash += trade.pnl

    def open_trade(self, num_shares, price, time_index, comment, stop_loss, take_profit):
        # TODO: handle all this in the ctor of Trade class? 
        
        new_trade = Trade(self, num_shares, price, time_index, comment)
        self.trades[comment] = new_trade
        
        # If originally a long trade, adjust the cash
        # If originally a short trade, "borrowed shares" are not cash, and covering does not change the cash, for simplicity. 
        if new_trade.num_shares > 0:
            self.cash -= price * new_trade.num_shares
        
        # TODO: implement trailing stop loss, modify the stop / limit price every tick
        if stop_loss:
            new_trade.stop_loss_order = self.new_order(-num_shares, stop_price=stop_loss, parent_trade=new_trade, comment=comment)
        if take_profit:
            new_trade.take_profit_order = self.new_order(-num_shares, limit_price=take_profit, parent_trade=new_trade, comment=comment)
    
    def last_price(self):
        return self.data.iloc[self._i].Close

# A middleman between OHLC signals and broker
class Strategy:
    def __init__(self, data, broker) -> None:
        self._i = 0
        self.data = data
        self.broker = broker
        self.indicators = set()
    
    def next(self, verbose=False):
        self._i += 1
        
    def set_index(self, new_index):
        self._i = new_index

    def buy(self, num_shares, limit_price = None, stop_loss = None, take_profit = None, comment = None):
        assert num_shares > 0, "num_shares must be > 0"
        return self.broker.new_order(num_shares, limit_price, stop_loss, take_profit, comment)
    
    def sell(self, num_shares, limit_price = None, stop_loss = None, take_profit = None, comment = None):
        assert num_shares > 0, "num_shares must be > 0"
        return self.broker.new_order(-num_shares, limit_price, stop_loss, take_profit, comment)
    
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
