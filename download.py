import os
import datetime as dt
import pandas as pd
import yfinance as yf

def _validate_args(ticker, interval, start, end):
    VALID_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","4h","1d","5d","1wk","1mo","3mo"}
    assert interval in VALID_INTERVALS, f"Invalid interval: {interval}. Supported intervals: {VALID_INTERVALS}"
    assert isinstance(start, dt.datetime), f"Invalid start date"
    assert isinstance(end, dt.datetime), f"Invalid end date"

def is_continuous(datetime_series, interval:str) -> bool: 
    # TODO: check out pandas-market-calendars
    if interval.endswith("m"):
        pass # TODO
    elif interval.endswith("h"):
        pass # TODO
    elif interval.endswith("d"):
        pass # TODO
    elif interval.endswith("wk"):
        raise NotImplementedError("not yet support this") # TODO
    elif interval.endswith("mo"):
        raise NotImplementedError("not yet support this") # TODO
    return False

def download_from_yf(ticker, interval, start, end):
    MAX_CHUNK_SIZE_DICT = {"1m": dt.timedelta(days=7),
                           "2m": dt.timedelta(days=7),
                           "5m": dt.timedelta(days=7),
                           "15m": dt.timedelta(days=7),
                           "30m": dt.timedelta(days=7),
                           "1h": dt.timedelta(days=730)}
    max_chunk_size = None
    if interval in MAX_CHUNK_SIZE_DICT:
        max_chunk_size = MAX_CHUNK_SIZE_DICT[interval]
    
    # TODO: call yf.download(), process index as "Datetime", and save to the right pkl file
    data = yf.download(ticker, interval=interval, start=start, end=end, prepost=True)
    return data
        

def get_ohlcv(ticker:str, interval:str, start:dt.datetime, end:dt.datetime):
    '''
    Note: 4h is special, gotta download it from somewhere other than yf
    '''
    _validate_args(ticker, interval, start, end)
    pickle_file = os.path.join(os.getcwd(), "data", f"{ticker.upper()}_{interval.lower()}.pkl")
    if os.path.exists(pickle_file):
        df = pd.read_pickle(pickle_file)
        filtered_df = df[(df['Datetime'] >= start) & (df['Datetime'] <= end)]
    if not is_continuous(filtered_df['Datetime'], interval):
        return download_from_yf(ticker, interval, start, end)
    else:
        return filtered_df

