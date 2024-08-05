from download import * 
from visualize import * 
import pandas as pd
from utils import *
from typing import * 

def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def EMA(values, n):
    return pd.Series(values).ewm(span=n, adjust=False).mean()

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
        super().next()
        if self.crossover(self.ema9, self.sma50):
            self.buy(1)

        elif self.crossover(self.sma50, self.ema9):
            self.sell(1)
        
data = get_historical_data("VOO", "1d")
broker = Broker(data)
strategy = SmaCross(data, broker)

# Skip first few candles where indicators are still "warming up"
# +1 to have at least two entries available
start = max((np.isnan(indicator.values.astype(float)).argmin(axis=-1).max()
                    for indicator in strategy.indicators), default=0)

broker.set_index(start)
strategy.set_index(start)

# This assumes decisions are made only at close of candle, and orders would always go thru at the open of next candle. 
# This backtest simulation cannot make new trades in the middle of the candle, which makes sense due to 1. the lack of repainting data, 2. the general advice that traders should wait for a candle's close to make decisions. 
# An exception is that, it is assumed that stop loss orders and take profit orders go thru immediately, even in the middle of a candle. While this ignores the effects of bid-ask spreads and broker execution delays, a backtesting simulation like this one cannot accurately quantify these implications. 

for i in range(start, len(data)):
    # update state machines and indicators
    strategy.next()
    
    try:
        broker.next()
    except OutOfMoneyError:
        break

# else:
#     for trade in broker.active_trades:
#         trade.close()
        

# compute equity and stats
visualize(data, broker, strategy)