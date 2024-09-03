import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay


MAX_NUM_YEARS = 50
BASE_DIR = "C:\\Users\\georg\\GitHub\\money-tree\\"

def _validate_args(ticker, interval, start, end):
    VALID_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","4h","1d","5d","1wk","1mo","3mo"}
    assert interval in VALID_INTERVALS, f"Invalid interval: {interval}. Supported intervals: {VALID_INTERVALS}"
    if start is not None:
        assert isinstance(start, dt.date), f"Invalid start date"
    if end is not None:
        assert isinstance(end, dt.date), f"Invalid end date"

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

        
def read_pickle(pickle_filepath):
    if os.path.exists(pickle_filepath):
        return pd.read_pickle(pickle_filepath)
    else:
        return None

def pickle_filepath(ticker, interval):
    return os.path.join(BASE_DIR, "data", f"{ticker.upper()}_{interval.lower()}.pkl")


def update_from_yf(ticker, interval, start: dt.datetime = None, end:dt.datetime = None, df:pd.DataFrame = pd.DataFrame(columns=["Datetime", 'Open', 'High', 'Low', 'Close', 'Volume'])):
    '''
    This function is expected to be called with all arguments defined, no arguments should equal to None. 
    '''
    # Call yf.download(), join on "Datetime", and save to pkl file
    data = yf.download(ticker, interval=interval, start=start, end=end, prepost=True)
    data = data.drop(['Adj Close'], axis=1)
    try:
        data.index = list(map(lambda x: dt.datetime.strptime(str(x).replace(":",""), '%Y-%m-%d %H%M%S%z').replace(tzinfo=None), data.index))
    except:
        try:
            data.index = list(map(lambda x: dt.datetime.strptime(str(x).replace(":",""), '%Y-%m-%d %H%M%S').replace(tzinfo=None), data.index))
        except Exception as e:
            raise ValueError(e)
    data = data.rename_axis("Datetime").reset_index()
    df = df.merge(data, how="outer")
    target_file = pickle_filepath(ticker, interval)
    if not os.path.exists(target_file):
        open(target_file, "x")
    df.to_pickle(target_file)
    return data


def get_historical_data(ticker:str, interval:str, start:dt.date = None, end:dt.date = None):
    '''
    Note: 4h is special, gotta download it from somewhere other than yf
    '''
    _validate_args(ticker, interval, start, end)

    df = read_pickle(pickle_filepath(ticker, interval))
    # Make start and end valid
    if start is None: 
        if df is not None: # If pickle exists
            start = df['Datetime'].min()
        else: # If pickle doesn't exist
            start = dt.datetime(dt.date.today().year - MAX_NUM_YEARS, 1, 1)
    if end is None: 
        end = dt.datetime.today()
    if type(start) == dt.date:
        start = dt.datetime(start.year, start.month, start.day)
    if type(end) == dt.date:
        end = dt.datetime(end.year, end.month, end.day)
    
    # Interval string to number of days
    MAX_DAYS_DICT = {"1m": 7, 
                    "2m": 7,
                    "5m": 7,
                    "15m": 7,
                    "30m": 7,
                    "1h": 730}
    num_days = MAX_DAYS_DICT.get(interval, np.inf)
    
    # For now assume data is downloaded by whole days， i.e. whole day present or whole day missing
    # TODO: remove this assumption and print a warning if there's missing timestamps
    # business_days = pd.bdate_range(str(start), str(end)) # (start.isoformat(), end.isoformat())
    
    if (dt.datetime.today() - end).days > num_days:
        raise ValueError(f"Cannot query {interval} data greater than {num_days} days!")
    
    if num_days != np.inf:
        start = max(start, dt.datetime.today() - dt.timedelta(days=num_days-1))
    
    if df is not None: # If pickle exists     
        filtered_df = df[(df['Datetime'] >= start) & (df['Datetime'] <= end)]
        if not is_continuous(filtered_df['Datetime'], interval):
            return update_from_yf(ticker, interval, start, end, df)
        else:
            return filtered_df
    
    else: # If pickle doesn't exist        
        df = pd.DataFrame(columns=["Datetime", 'Open', 'High', 'Low', 'Close', 'Volume'])
        return update_from_yf(ticker, interval, start, end, df)


# in case there's stock split or something that overwrites the entire history, or in case something goes wrong
def overwrite_historic_data(ticker, interval, df):
    if df is None: 
        raise RuntimeError("No data to overwrite!")
    start = df['Datetime'].min()
    end = dt['Datetime'].max()
    return update_from_yf(ticker, interval, start, end, df)

def update_all():
    for interval in {"1m", "5m", "1h", "1d"}:
        for ticker in {"QQQ", "SPY", "VOO", "NVDA", "MSFT"}:
            get_historical_data(ticker, interval)