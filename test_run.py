from download import * 
from visualize import * 
import pandas as pd
from utils import *
from typing import * 

def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def EMA(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean()

def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)

class OutOfMoneyError(Exception):
    pass

class SmaCross(Strategy):   
    def __init__(self, data, broker) -> None:
        super().__init__(data, broker)
        
        # instantiate indicators
        self.sma50 = Indicator(SMA, self.data.Close, 50, overlay=True)
        self.ema9 = Indicator(EMA, self.data.Close, 9, overlay=True)
        
        # all indicators used in strategy
        self.indicators = {indicator
                    for _, indicator in self.__dict__.items()
                    if isinstance(indicator, Indicator)}

    def next(self):
        if self.crossover(self.ema9, self.sma50):
            self.buy(1)

        elif self.crossover(self.sma50, self.ema9):
            self.sell(1)

        super().next()

data = get_historical_data("NVDA", "1m")
broker = Broker(data)
strategy = SmaCross(data, broker)


# Input Validation (skip for now). 
# TODO: wrap this script in a backtest class and add these validations
# if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
#             raise TypeError('`strategy` must be a Strategy sub-type')
# if not isinstance(data, pd.DataFrame):
#     raise TypeError("`data` must be a pandas.DataFrame with columns")
# if len(data) == 0:
#     raise ValueError('OHLC `data` is empty')
# if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
#     raise ValueError("`data` must be a pandas.DataFrame with columns "
#                       "'Open', 'High', 'Low', 'Close', and 'Volume'")
# if data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().values.any():
#     raise ValueError('Some OHLCV values are missing (NaN). '
#                       'Please strip those lines with `df.dropna()` or '
#                       'fill them in with `df.interpolate()`.')        


# Skip first few candles where indicators are still "warming up"
# +1 to have at least two entries available
start = int(max((np.isnan(indicator.values.astype(float)).argmin(axis=-1).max()
                    for indicator in strategy.indicators), default=1))
broker.set_index(start)
strategy.set_index(start)

# This assumes decisions are made only at close of candle, and orders would always go thru at the open of next candle. 
# This backtest simulation cannot make new trades in the middle of the candle, which makes sense due to 1. the lack of repainting data, 2. the general advice that traders should wait for a candle's close to make decisions. 
# An exception is that, it is assumed that stop loss orders and take profit orders go thru immediately, even in the middle of a candle. While this ignores the effects of bid-ask spreads and broker execution delays, a backtesting simulation like this one cannot accurately quantify these implications. 

for i in range(start, len(data)):
    # update state machines and indicators
    # NOTE: The first tick of the for loop looks at index 1 for both strategy and broker
    strategy.next()
    
    try:
        broker.next()
    except OutOfMoneyError:
        break

# else:
#     # Close any remaining open trades so they produce some stats
#     for trade in broker.active_trades:
#         trade.close()
        

# compute equity and stats
# visualize(data, broker, strategy)

# update_all()