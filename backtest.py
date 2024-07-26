# import datetime as dt
# import numpy as np
# from download import *
# from strategy import *

# def backtest(strategy : Strategy, ticker : str, interval : str, portfolio : Portfolio, start : dt.datetime = None, end: dt.datetime = None):
#     '''
#     a list of transaction (buy/sell) history
#     each transaction has name, datetime, size, price
#     '''
#     # TODO: test types 
#     # if not issubclass(strategy, Strategy):
#     #     raise TypeError("strategy must be a Strategy sub-type")
    
#     # download if incomplete
#     historic_data = get_ohlcv(ticker, interval, start, end)
    
#     # TODO: fault handling of data fetching
   
#     strategy.initialize()
    
#     for ix, data in enumerate(historic_data):
        
#         # No decisions if indicators that are NaN (warming up)
#         # NOTE: When downloading make sure the previous day data is present, so to ensure the first tick of the day has full indicators. 
#         if any(np.isnan(indicator) for indicator in strategy.indicators):
#             continue
        
#         portfolio.next()
#         strategy.next()
        
        
#     else:
#         # Close any remaining open trades so they produce some stats
#         for trade in portfolio.trades:
#             trade.close()

#     results = compute_results(trades=portfolio.closed_trades,
#                 equity=pd.Series(portfolio.equity).bfill().fillna(portfolio.cash).values,
#                 ohlc_data=historic_data)
#     return results

#     # strategy manages the protfolio? 
#     # print srat info e.g. $ avg win, % win, $ avg loss, % loss
#     # plot strat returns
#     # plot candlestick history with entry exit points
    
#     '''
#     decision class has objects: 
#         action
#         amount
#         stop loss (price or percentage object, a validated string with or without % at end)
#         profit target (same as above)
#     '''
#     # decision class has objects: action, amount, stop loss, profit target
#     # execute function handles impact of buy / sell / SL / take profit etc on portfolio
#     # sell all by EOD or sell all by end date


# def compute_results(trades,
#                 equity,
#                 ohlc_data):
#     pass