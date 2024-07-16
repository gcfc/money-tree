def backtest(strategy, ticker : str, interval : str, start : dt.datetime = None, end: dt.datetime = None):
    '''
    a list of transaction (buy/sell) history
    each transaction has name, datetime, size, price
    '''
     # download if incomplete
    historic_data = get_ohlcv(ticker, interval, start, end)
    trading_session = TradingSession()
    for ix, data in enumerate(historic_data):
        decision = strategy.process(data)
        trading_session.execute(decision)
    # strategy manages the protfolio? 
    # print srat info e.g. $ avg win, % win, $ avg loss, % loss
    # plot strat returns
    # plot candlestick history with entry exit points
    
    '''
    decision class has objects: 
        action
        amount
        stop loss (price or percentage object, a validated string with or without % at end)
        profit target (same as above)
    '''
    # decision class has objects: action, amount, stop loss, profit target
    # execute function handles impact of buy / sell / SL / take profit etc on portfolio
    # sell all by EOD or sell all by end date