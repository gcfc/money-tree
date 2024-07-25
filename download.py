import os
import datetime as dt
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


MAX_NUM_YEARS = 50

def _validate_args(ticker, interval, start, end):
    VALID_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h","4h","1d","5d","1wk","1mo","3mo"}
    assert interval in VALID_INTERVALS, f"Invalid interval: {interval}. Supported intervals: {VALID_INTERVALS}"
    assert isinstance(start, dt.date), f"Invalid start date"
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
    return os.path.join(os.getcwd(), "data", f"{ticker.upper()}_{interval.lower()}.pkl")

def split_date_range_to_query(start_date, end_date, max_chunk_size):
    if max_chunk_size is None:
        return [(start_date, end_date)]
    
    # Convert to pandas Timestamps if they are not already
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Initialize the list to hold the date ranges
    date_ranges = []
    
    # Loop through the date range
    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + BDay(max_chunk_size)
        
        # Ensure the end date does not go beyond the overall end_date
        if current_end > end_date:
            current_end = end_date
        
        # Append the tuple of the current start and end date to the list
        date_ranges.append((current_start.date(), current_end.date()))
        
        # Move to the next start date
        current_start = current_end + BDay(1)
    
    return date_ranges


def update_from_yf(ticker, interval, start, end, df):
    # Interval string to number of days
    MAX_CHUNK_SIZE_DICT = {"1m": 7, 
                           "2m": 7,
                           "5m": 7,
                           "15m": 7,
                           "30m": 7,
                           "1h": 730}
    max_chunk_size = MAX_CHUNK_SIZE_DICT.get(interval)
    
    # For now assume data is downloaded by whole days
    # TODO: remove this assumption and print a warning if there's missing timestamps
    business_days = pd.bdate_range(str(start), str(end)) # (start.isoformat(), end.isoformat())
    
    # blob dates into max_chunk_size and download
    # TODO: only download the missing dates, not all from start to end
    queries_list = split_date_range_to_query(start, end, max_chunk_size)
        
    for query_start, query_end in queries_list:
        # Call yf.download() by max chunks, join on "Datetime", and save to pkl file
        data = yf.download(ticker, interval=interval, start=query_start, end=query_end, prepost=True)
        df = pd.merge(df, data, on="Datetime", how="left") # TODO verify this
        df.to_pickle(pickle_filepath(ticker, interval)) # TODO: verify this
    return data

        
def get_ohlcv(ticker:str, interval:str, start:dt.date, end:dt.date):
    '''
    Note: 4h is special, gotta download it from somewhere other than yf
    '''
    _validate_args(ticker, interval, start, end)
    
    df = read_pickle(pickle_filepath(ticker, interval))
    
    # If pickle exists
    if df is not None:
        if start is None: 
            start = df['Datetime'].min()
        if end is None: 
            end = dt.date.today()
        filtered_df = df[(df['Datetime'] >= start) & (df['Datetime'] <= end)]
    
        if not is_continuous(filtered_df['Datetime'], interval):
            return update_from_yf(ticker, interval, start, end, df)
        else:
            return filtered_df
    
    # If pickle doesn't exist
    else:
        if start is None: 
            dt.date(dt.date.today().year - MAX_NUM_YEAR, 1, 1)
        if end is None: 
            end = dt.date.today()
        return update_from_yf(ticker, interval, start, end, df)


# in case there's stock split or something that overwrites the entire history, or in case something goes wrong
def overwrite_historic_data(ticker, interval, df):
    if df is None: 
        raise RuntimeError("No data to overwrite!")
    start = df['Datetime'].min()
    end = dt['Datetime'].max()
    return update_from_yf(ticker, interval, start, end, df)
