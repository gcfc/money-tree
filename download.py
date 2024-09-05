import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay
from typing import *
import requests
from dotenv import load_dotenv
load_dotenv()


BEGINNING_OF_TIME = dt.date(1960, 1, 1)
BASE_DIR = sys.path[0]
VALID_INTERVALS = {"1m","2m","5m","15m","30m","1h","4h","1d"} # "60m","90m","5d","1wk","1mo","3mo"
# TODO: to be used in is_continuous
INTERVALS_TO_TIMEDELTA = {
    "1m": dt.timedelta(minutes=1),
    "2m": dt.timedelta(minutes=2),
    "5m": dt.timedelta(minutes=5),
    "15m": dt.timedelta(minutes=15),
    "30m": dt.timedelta(minutes=30),
    "1h": dt.timedelta(hours=1),
    "4h": dt.timedelta(hours=4),
    "1d": dt.timedelta(days=1)
}
MARKET_OPEN = pd.to_datetime('09:30:00').time()
MARKET_CLOSE = pd.to_datetime('16:00:00').time()

# Interval string to number of days
MAX_DAYS_DICT_YF = {"1m": 7, 
                    "2m": 7,
                    "5m": 7,
                    "15m": 7,
                    "30m": 7,
                    "1h": 730}
MAX_DAYS_DICT_POLYGON = {interval : 730 for interval in VALID_INTERVALS}

def _validate_args(ticker, interval, start, end):
    assert interval in VALID_INTERVALS, f"Invalid interval: {interval}. Supported intervals: {VALID_INTERVALS}"
    if start is not None:
        assert type(start) == dt.date, f"Invalid start date, must be type datetime.date"
    if end is not None:
        assert type(end) == dt.date, f"Invalid end date, must be type datetime.date"

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


def download_from_yf(ticker, interval, start: dt.date, end:dt.date):
    '''
    This function is expected to be called with all arguments defined, no arguments should equal to None.
    Downloads data from yfinance and returns a cleaned-up pandas DataFrame.  
    '''
    data = yf.download(ticker, interval=interval, start=start, end=end, prepost=True, progress=False)
    data = data.drop(['Adj Close'], axis=1)
    try:
        data.index = list(map(lambda x: dt.datetime.strptime(str(x).replace(":",""), '%Y-%m-%d %H%M%S%z').replace(tzinfo=None), data.index))
    except:
        try:
            data.index = list(map(lambda x: dt.datetime.strptime(str(x).replace(":",""), '%Y-%m-%d %H%M%S').replace(tzinfo=None), data.index))
        except Exception as e:
            raise ValueError(e)
    data = data.rename_axis("Datetime").reset_index()
    return data

def download_from_polygon(ticker, interval, start: dt.date, end:dt.date):
    '''
    This function is expected to be called with all arguments defined, no arguments should equal to None.
    Downloads data from polygon and returns a cleaned-up pandas DataFrame.  
    '''
    interval_abbrev_dict = {"m": "minute", "h": "hour", "d": "day"}
    interval_arg = interval[:-1] + "/" + interval_abbrev_dict[interval[-1]]
    
    polygon_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{interval_arg}/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=50000&apiKey={os.environ['POLYGON_API_KEY']}"
    r = requests.get(polygon_url)
    if not (r.json() and "results" in r.json() and "status" in r.json() and r.json()["status"] == "OK"):
        raise RuntimeError(f"Polygon JSON Error! {r.json()}")
    results = r.json()['results']
    df = pd.json_normalize(results)
    df.t = pd.Series.apply(df.t, lambda x: dt.datetime.fromtimestamp(x*1E-3).astimezone(dt.timezone(dt.timedelta(hours=-4))).replace(tzinfo=None))
    df = df.rename(columns={"t":"Datetime", "o": "Open", "h":"High", "l":"Low",  "c": "Close", "v": "Volume"})
    df = df.drop(['vw', 'n'], axis=1)
    return df

def get_historical_data(ticker:str, 
                        interval:str, 
                        start:Union[dt.date, None] = None, 
                        end:Union[dt.date, None] = None,
                        save_to_pickle = True):
    '''
    The public interface to get historical data from any time at any intervals.
    This function fuses together many sources to produce a cleaned-up pandas DataFrame 
    with all the data.
    '''
    _validate_args(ticker, interval, start, end)

    max_days_yf = MAX_DAYS_DICT_YF.get(interval, np.inf)
    max_days_polygon = MAX_DAYS_DICT_POLYGON.get(interval, np.inf)
    max_days = max(max_days_yf, max_days_polygon)
    
    # Make start and end valid
    if start is None: 
        start = BEGINNING_OF_TIME
    if end is None: 
        end = dt.date.today()

    if max_days != np.inf:
        start = max(start, dt.date.today() - dt.timedelta(days=max_days-1))
    
    if (dt.date.today() - end).days > max_days:
        raise ValueError(f"Cannot query {interval} data greater than {max_days} days!")

    # At this point start and end date must be specified (i.e. not None)
    assert end > start, "Start date must be earlier than end date!"
        
    # For now assume data is downloaded by whole days， i.e. whole day present or whole day missing
    # TODO: remove this assumption and print a warning if there's missing timestamps
    # business_days = pd.bdate_range(str(start), str(end)) # (start.isoformat(), end.isoformat())
    
    new_data = None
    if interval == "1d":
        new_data = download_from_yf(ticker, interval, start, end)
    elif interval == "4h":
        new_data = download_from_polygon(ticker, interval, start, end)
    else:
        # Fuse the two together. During pre- and post- market hours, yfinance has better time granularity but no volume data, whereas Polygon has sparse timestamps each with volume data. 
        # TODO: test call doesn't work df = get_historical_data("EBS", "1m")
        new_data_yf = download_from_yf(ticker, interval, start, end)
        new_data_polygon = download_from_polygon(ticker, interval, start, end)
        
        pre_post_data = new_data_polygon.loc[(new_data_polygon['Datetime'].dt.time <= MARKET_OPEN) | (new_data_polygon['Datetime'].dt.time >= MARKET_CLOSE)]

        # TODO: Lookup not by exact value, but by the closest timestamp in new_data_yf after the query timestamp. Polygon and yfinance have almost the exact timestamps so it doesn't quite matter for a prototype. 
        
        # Populate pre- post- market volume
        for _, row in pre_post_data.iterrows():
            yf_row = new_data_yf[new_data_yf['Datetime'] == row.Datetime]
            assert len(yf_row) <= 1, "Multiple of the same timestamps found."
            new_data_yf.loc[yf_row.index, 'Volume'] = row.Volume
        
        new_data = new_data_yf

    if not save_to_pickle:
        return new_data
    else:
        df = read_pickle(pickle_filepath(ticker, interval))
        merge = False
        if df is not None: # If pickle exists     
            filtered_df = df[(df['Datetime'] >= start) & (df['Datetime'] <= end)]
            if not is_continuous(filtered_df['Datetime'], interval):
                merge = True
            else:
                return filtered_df
        
        else: # If pickle doesn't exist        
            df = pd.DataFrame(columns=["Datetime", 'Open', 'High', 'Low', 'Close', 'Volume'])
            merge = True
        
        if merge:
            df = df.merge(new_data, how="outer")
        target_file = pickle_filepath(ticker, interval)
        if not os.path.exists(target_file):
            open(target_file, "x")
        df.to_pickle(target_file)
        return df


# in case there's stock split or something that overwrites the entire history, or in case something goes wrong
def overwrite_historic_data(ticker, interval, df):
    if df is None: 
        raise RuntimeError("No data to overwrite!")
    start = df['Datetime'].min()
    end = dt['Datetime'].max()
    return download_from_yf(ticker, interval, start, end, df)

def update_all():
    for interval in {"1m", "5m", "1h", "1d"}:
        for ticker in {"QQQ", "SPY", "VOO", "NVDA", "MSFT"}:
            get_historical_data(ticker, interval)
