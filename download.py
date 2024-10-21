import yfinance as yf
from utils import *
from typing import *
import requests
import time
from bisect import bisect_left
import pandas_market_calendars as mcal

BEGINNING_OF_TIME = dt.date(1960, 1, 1)

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
                    "2m": 60,
                    "5m": 60,
                    "15m": 60,
                    "30m": 60,
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
    assert type(start) == dt.date, f"Invalid start date, must be type datetime.date"
    assert type(end) == dt.date, f"Invalid end date, must be type datetime.date"

def modify_start_end_by_downloader(interval, start, end):
    assert start is not None, "Start date not specified to modify!"
    assert end is not None, "End date not specified to modify!"
    output = {"yf": (None, None), "polygon": (None, None)}
    max_days_yf = MAX_DAYS_DICT_YF.get(interval, np.inf)
    max_days_polygon = MAX_DAYS_DICT_POLYGON.get(interval, np.inf)
    
    yf_start = max(start, dt.date.today() - dt.timedelta(days=max_days_yf+1))
    if not (yf_start > end):
        output["yf"] = (yf_start, end)

    polygon_start = max(start, dt.date.today() - dt.timedelta(days=max_days_polygon))
    if not (polygon_start > end):
        output["polygon"] = (polygon_start, end)

    assert type(output) == dict and len(output) == 2 and all(len(val) == 2 for val in output.values()) 
    return output

def is_strictly_continuous(datetime_series, interval:str) -> tuple: 
    # Must match all days and every timestamp
    if len(datetime_series) == 0:
        return True, []
    expected_times = []
    start = datetime_series.min().normalize()  
    end = datetime_series.max().normalize()
    market_schedule = mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end)
    business_days_list = market_schedule.loc[market_schedule['market_close'].dt.time >= pd.to_datetime('20:00:00').time()].index.to_list()
    for date in business_days_list:
        if interval == '1d':
            expected_times.append(pd.date_range(start=pd.Timestamp(date).normalize(), end=pd.Timestamp(date).normalize(), freq=INTERVALS_TO_PD_OFFSET[interval]))

        else:
            # All stock charts should have candlestick that starts at 9:30am each day
            # TODO: Forex may have different timestamps. Modify this if I ever touch Forex. Maybe pass in the ticker symbol and check if it ends with "=X". If so day_start and day_end are just the .normalize() 1 day apart (i.e. all day long on the hour). 
            day_start = pd.Timestamp(date).normalize() + pd.Timedelta(hours=9, minutes=30)
            if interval == '4h':
                day_start = pd.Timestamp(date).normalize() + pd.Timedelta(hours=4)
            day_end = pd.Timestamp(date).normalize() + pd.Timedelta(hours=16)
            day_times = [i.to_datetime64() for i in pd.date_range(start=day_start, end=day_end, freq=INTERVALS_TO_PD_OFFSET[interval])]
            expected_times.extend(day_times)
    missing_times = np.setdiff1d(expected_times, datetime_series.values)

    return (len(missing_times) == 0), missing_times

def is_loosely_continuous(datetime_series, interval:str) -> tuple:
    # as long as the 9:30am candle is downloaded that day, assume the whole day is downloaded
    if len(datetime_series) == 0:
        return True, []
    start = datetime_series.min().normalize()  
    end = datetime_series.max().normalize()
    market_schedule = mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end)
    business_days_list = market_schedule.loc[market_schedule['market_close'].dt.time >= pd.to_datetime('20:00:00').time()].index.to_list()
    if interval == '1d':
        expected_times = pd.date_range(start=start, end=end)
        missing_times = np.setdiff1d(expected_times, datetime_series)
        return (len(missing_times) == 0), missing_times
    everyday_930 = [day.replace(hour=9, minute=30).to_datetime64() for day in business_days_list]
    missing_days = [bd for bd in everyday_930 if bd not in datetime_series.values]
    return (len(missing_days) == 0), missing_days


def most_recent_business_day(query_date = dt.date.today()):
    ind = bisect_left(FULL_BUSINESS_DAYS, query_date)
    return query_date if (query_date == FULL_BUSINESS_DAYS[ind]) else FULL_BUSINESS_DAYS[ind-1]

def download_from_yf(ticker, interval, start: Union[dt.date,None], end: Union[dt.date,None]):
    '''
    This function is expected to be called with either valid arguments defined, or both start and end equal to None. 
    Downloads data from yfinance and returns a cleaned-up pandas DataFrame.  
    If error, return empty pandas DataFrame with correct column titles.
    '''

    data = pd.DataFrame(columns=["Datetime", 'Open', 'High', 'Low', 'Close', 'Volume'])
    if start is None and end is None:
        return data
    _validate_args(ticker, interval, start, end)
    
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

def download_from_polygon(ticker, interval, start: Union[dt.date,None], end: Union[dt.date,None]):
    '''
    This function is expected to be called with either valid arguments defined, or both start and end equal to None. 
    Downloads data from polygon and returns a cleaned-up pandas DataFrame. 
    If error, return empty pandas DataFrame with correct column titles.
    '''
    total_df = pd.DataFrame(columns=["Datetime", 'Open', 'High', 'Low', 'Close', 'Volume'])  
    if start is None and end is None:
        return total_df
    _validate_args(ticker, interval, start, end)
    
    interval_abbrev_dict = {"m": "minute", "h": "hour", "d": "day"}
    interval_arg = interval[:-1] + "/" + interval_abbrev_dict[interval[-1]]

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
        
    total_df = total_df.rename(columns={"t":"Datetime", "o":"Open", "h":"High", "l":"Low", "c": "Close", "v":"Volume"})
    total_df = total_df.drop(['vw', 'n'], axis=1)
    return total_df

def download_and_save(ticker:str, 
                    interval:str, 
                    start:Union[dt.date, None] = None, 
                    end:Union[dt.date, None] = None,
                    save_to_pickle = True,
                    allow_low_quality = False):
    '''
    The public interface to get historical data from any time at any intervals.
    This function fuses together many sources to produce a cleaned-up pandas DataFrame 
    with all the data.
    start = None means download from beginning of time
    end = None means download until and including today
    '''
    if start is None: 
        start = BEGINNING_OF_TIME
    if end is None: 
        end = dt.date.today()
    _validate_args(ticker, interval, start, end)

    # Check existing data and either 1. modify start and end or 2. don't download at all. 
    # TODO: save_to_pickle can be False and can still access and return existing data, just don't modify existing data
    if save_to_pickle:
        df = get_downloaded_data_or_none(ticker, interval, start, end)
        if df is not None and len(df) > 0: # If data is already downloaded
            is_df_continuous, missing_times = is_loosely_continuous(df['Datetime'], interval)
            if is_df_continuous:
                return df
            start = pd.Timestamp(missing_times[0]).to_pydatetime().date()
    
    print(f"Downloading:\t{ticker}\t{interval}\t{start}\t{end}")
    new_data = None
    modified_starts_and_ends = modify_start_end_by_downloader(interval, start, end)
    if interval == "1d":
        new_data = download_from_yf(ticker, interval, *modified_starts_and_ends["yf"])
    elif interval == "4h":
        new_data = download_from_polygon(ticker, interval, *modified_starts_and_ends["polygon"])
    else:
        # Fuse the two together. During pre- and post- market hours, yfinance has better time granularity but no volume data, whereas Polygon has sparse timestamps each with volume data. 

        new_data_yf = download_from_yf(ticker, interval, *modified_starts_and_ends["yf"])
        new_data_polygon = download_from_polygon(ticker, interval, *modified_starts_and_ends["polygon"])
        
        if (len(new_data_yf) == 0 and len(new_data_polygon) == 0):
            new_data = pd.DataFrame(columns=["Datetime", 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        elif (len(new_data_yf) > 0 and len(new_data_polygon) > 0):
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
                
        elif allow_low_quality:
            if len(new_data_polygon) > 0:
                new_data = new_data_polygon
            elif len(new_data_yf) > 0:
                new_data = new_data_yf           

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
            download_and_save(ticker, interval, end=dt.date.today()-dt.timedelta(days=1))
