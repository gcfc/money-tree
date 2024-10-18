import os
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay
from typing import *
import requests
import time
from bisect import bisect_left
import pandas_market_calendars as mcal
from dotenv import load_dotenv
load_dotenv()

BEGINNING_OF_TIME = dt.date(1960, 1, 1)
BASE_DIR = os.path.join(os.environ["BASE_DIR"])

VALID_INTERVALS = {"1m","2m","5m","15m","30m","1h","4h","1d"} # "60m","90m","5d","1wk","1mo","3mo"
# NOTE: "4h" is more useful for forex & crypto. Download the data for now, but note that candle starts at 8am instead of 9:30am. 

INTERVALS_TO_PD_OFFSET = {
    "1m": '1min',
    "2m": '2min',
    "5m": '5min',
    "15m": '15min',
    "30m": '30min',
    "1h": '1h',
    "4h": '4h',
    "1d": '1D'
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

FULL_BUSINESS_DAYS = None
BUSINESS_DAYS_UPDATED = False

def update_business_days_array():
    global FULL_BUSINESS_DAYS, BUSINESS_DAYS_UPDATED
    last_year_start = dt.date(dt.date.today().year - 1, 1, 1)
    next_year_end = dt.date(dt.date.today().year + 1, 12, 31)
    market_schedule = mcal.get_calendar('NYSE').schedule(start_date=str(last_year_start), end_date=str(next_year_end))
    business_days_list = market_schedule.loc[market_schedule['market_close'].dt.time >= pd.to_datetime('20:00:00').time()].index.to_list()
    FULL_BUSINESS_DAYS = [i.to_pydatetime().date() for i in business_days_list]
    BUSINESS_DAYS_UPDATED = True
    print("Business days loaded!")

update_business_days_array()

def _validate_args(ticker, interval, start, end):
    assert interval in VALID_INTERVALS, f"Invalid interval: {interval}. Supported intervals: {VALID_INTERVALS}"
    if start is not None:
        assert type(start) == dt.date, f"Invalid start date, must be type datetime.date"
    if end is not None:
        assert type(end) == dt.date, f"Invalid end date, must be type datetime.date"

def is_continuous(datetime_series, interval:str) -> bool: 
    expected_times = []
    start = datetime_series.min().normalize()  
    end = datetime_series.max().normalize()
    market_schedule = mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end)
    business_days_list = market_schedule.loc[market_schedule['market_close'].dt.time >= pd.to_datetime('20:00:00').time()].index.to_list()
    if interval == '1d':
        expected_times = pd.date_range(start=start, end=end)
    for date in business_days_list:
        if interval == '4h':
            day_start = pd.Timestamp(date).normalize() + pd.Timedelta(hours=4)
        else:
            # All stock charts should have candlestick that starts at 9:30am each day
            # TODO: Forex may have different timestamps. Modify this if I ever touch Forex. Maybe pass in the ticker symbol and check if it ends with "=X". If so day_start and day_end are just the .normalize() 1 day apart (i.e. all day long on the hour). 
            day_start = pd.Timestamp(date).normalize() + pd.Timedelta(hours=9, minutes=30)
        day_end = pd.Timestamp(date).normalize() + pd.Timedelta(hours=16)
        day_times = pd.date_range(start=day_start, end=day_end, freq=INTERVALS_TO_PD_OFFSET[interval])
        expected_times.append(day_times)

    missing_times = np.setdiff1d(expected_times, datetime_series)

    return (len(missing_times) == 0), missing_times

def read_pickle(pickle_filepath):
    if os.path.exists(pickle_filepath):
        return pd.read_pickle(pickle_filepath)
    else:
        return None

def pickle_filepath(ticker, interval):
    return os.path.join(BASE_DIR, "data", f"{ticker.upper()}_{interval.lower()}.pkl")

def most_recent_business_day(query_date = dt.date.today()):
    ind = bisect_left(FULL_BUSINESS_DAYS, query_date)
    return query_date if (query_date == FULL_BUSINESS_DAYS[ind]) else FULL_BUSINESS_DAYS[ind-1]

def download_from_yf(ticker, interval, start: Union[dt.date, None], end: Union[dt.date, None]):
    '''
    This function is expected to be called with all arguments defined, no arguments should equal to None.
    Downloads data from yfinance and returns a cleaned-up pandas DataFrame.  
    '''
    # Modify start and end to valid range
    # TODO: extract this section from two functions and merge into its own function? 
    max_days_yf = MAX_DAYS_DICT_YF.get(interval, np.inf)

    if max_days_yf != np.inf:
        start = max(start, dt.date.today() - dt.timedelta(days=max_days_yf-1))
    
    if (dt.date.today() - end).days > max_days_yf:
        raise ValueError(f"Cannot query {interval} data greater than {max_days_yf} days!")

    # At this point start and end date must be specified (i.e. not None)
    assert end >= start, f"Start date ({start}) must be earlier than end date ({end})!"
    
    # For YF, the end date is not inclusive but I'd like to design my API to be inclusive of start and end dates (consistent with Polygon). i.e. to download one day of data, you'd call download_from_yf(ticker, interval, start=<same_date>, end=<same_date>).
    end = dt.date(end.year, end.month, end.day) + dt.timedelta(days=1)

    # Download and clean
    data = yf.download(ticker, interval=interval, start=start, end=end, prepost=True, progress=False)
    
    data = data.drop(['Adj Close'], axis=1)
    data = data.rename_axis("Datetime").reset_index()
    
    if len(data) == 0:
        print("WARNING: Empty YF query!")
        return data
    
    try:
        data.Datetime = list(map(lambda x: dt.datetime.strptime(str(x).replace(":",""), '%Y-%m-%d %H%M%S%z').replace(tzinfo=None), data.Datetime))
    except:
        try:
            data.Datetime = list(map(lambda x: dt.datetime.strptime(str(x).replace(":",""), '%Y-%m-%d %H%M%S').replace(tzinfo=None), data.Datetime))
        except Exception as e:
            raise ValueError(e)

    return data

def download_from_polygon(ticker, interval, start: Union[dt.date, None], end: Union[dt.date, None]):
    '''
    This function is expected to be called with all arguments defined, no arguments should equal to None.
    Downloads data from polygon and returns a cleaned-up pandas DataFrame.  
    '''
    # Modify start and end to valid range
    # TODO: extract this section from two functions and merge into its own function? 
    max_days_polygon = MAX_DAYS_DICT_POLYGON.get(interval, np.inf)

    if max_days_polygon != np.inf:
        start = max(start, dt.date.today() - dt.timedelta(days=max_days_polygon-1))
    
    if (dt.date.today() - end).days > max_days_polygon:
        raise ValueError(f"Cannot query {interval} data greater than {max_days_polygon} days!")

    # At this point start and end date must be specified (i.e. not None)
    assert end >= start, f"Start date ({start}) must be earlier than end date ({end})!"
    
    # Download and clean
    interval_abbrev_dict = {"m": "minute", "h": "hour", "d": "day"}
    interval_arg = interval[:-1] + "/" + interval_abbrev_dict[interval[-1]]
    
    total_df = None  
    count = np.inf

    while count >= 50000: # Polygon limit number of candlesticks per query
        polygon_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{interval_arg}/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=50000&apiKey={os.environ['POLYGON_API_KEY']}"
        r = requests.get(polygon_url)
        if not (r.json() and "results" in r.json()):
            raise RuntimeError(f"Polygon JSON Error! {r.json()}")
        results = r.json()['results']
        df = pd.json_normalize(results)
        df.t = pd.Series.apply(df.t, lambda x: dt.datetime.fromtimestamp(x*1E-3).astimezone(dt.timezone(dt.timedelta(hours=-4))).replace(tzinfo=None))
        total_df = pd.concat([total_df, df], ignore_index=True)
        total_df = total_df.drop_duplicates("t").sort_values(by="t")
        df_end_datetime = total_df.t.iloc[len(total_df)-1]
        start = df_end_datetime.date()
        count = r.json()['count']
        time.sleep(12.25) # Obey limit of 5 API calls per minute
        
    total_df = total_df.rename(columns={"t":"Datetime", "o": "Open", "h":"High", "l":"Low",  "c": "Close", "v": "Volume"})
    total_df = total_df.drop(['vw', 'n'], axis=1)
    return total_df

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
    if start is None: 
        start = BEGINNING_OF_TIME
    if end is None: 
        end = dt.date.today()

    # Check existing data and either 1. modify start and end or 2. don't download at all. 
    if save_to_pickle:
        df = read_pickle(pickle_filepath(ticker, interval))
        if df is not None: # If pickle exists     
            filtered_df = df[(df['Datetime'].dt.date >= start) & (df['Datetime'].dt.date <= end)]
            if len(filtered_df) > 0:
                is_df_continuous, missing_times = is_continuous(filtered_df['Datetime'], interval)
                if is_df_continuous:
                    return filtered_df
                start = pd.Timestamp(missing_times[0]).to_pydatetime().date()
    
    print(f"Downloading:\t{ticker}\t{interval}\t{start}\t{end}")
    new_data = None
    if interval == "1d":
        new_data = download_from_yf(ticker, interval, start, end)
    elif interval == "4h":
        new_data = download_from_polygon(ticker, interval, start, end)
    else:
        # Fuse the two together. During pre- and post- market hours, yfinance has better time granularity but no volume data, whereas Polygon has sparse timestamps each with volume data. 

        new_data_yf = download_from_yf(ticker, interval, start, end)
        new_data_polygon = download_from_polygon(ticker, interval, start, end)
        
        if interval == '1h': 
            # Keep YF's timestamps which are on the hour pre- and post- market, and on 30th minutes during market hours. Fill in only the pre- and post- market volume from Polygon. This ensures a separate candlestick when market opens at 9:30am. 
            
            polygon_pre_post_data = new_data_polygon.loc[(new_data_polygon['Datetime'].dt.time <= MARKET_OPEN) | (new_data_polygon['Datetime'].dt.time >= MARKET_CLOSE)]

            # fill in volume here and produce new_data
            # Lookup by exact value. Polygon and yfinance have almost the exact timestamps so it doesn't quite matter for a prototype. 
            # Low-prio TODO: make it better
        
            # Populate pre- post- market volume from Polygon
            for _, row in polygon_pre_post_data.iterrows():
                yf_row = new_data_yf[new_data_yf['Datetime'] == row.Datetime]
                assert len(yf_row) <= 1, "Multiple of the same timestamps found."
                new_data_yf.loc[yf_row.index, 'Volume'] = row.Volume

            new_data = new_data_yf
        
        else:
            # Stitched data kinda makes sense as an appoximation of pre- and post- market blips with volume readings. Polygon and yfinance have almost the exact timestamps so it doesn't quite matter for a prototype. 
            # Fill in finer-grain timestamps from YF. Polygon data takes precedence, hence the order in the list.
            new_data = pd.concat([new_data_polygon, new_data_yf], axis=0)
            new_data = new_data.drop_duplicates("Datetime").sort_values(by="Datetime")

    if save_to_pickle:
        if df is None: # If pickle doesn't exist        
            df = pd.DataFrame(columns=["Datetime", 'Open', 'High', 'Low', 'Close', 'Volume'])

        df = df.merge(new_data, how="outer")
        target_file = pickle_filepath(ticker, interval)
        if not os.path.exists(target_file):
            open(target_file, "x")
        df.to_pickle(target_file)
    
    return new_data

def update_big_names_data():
    for interval in VALID_INTERVALS:
        for ticker in {"QQQ", "SPY", "VOO", "NVDA", "MSFT"}:
            get_historical_data(ticker, interval)
