from tradingview_screener import *
import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal
import time
import pickle
import os
from download import *

market_schedule = mcal.get_calendar('NYSE').schedule(start_date='2024-01-01', end_date='2024-12-31')
regular_business_days = market_schedule.loc[market_schedule['market_close'].dt.time >= pd.to_datetime('20:00:00').time()].index.to_list()
REGULAR_BUSINESS_DAYS = [i.to_pydatetime().date() for i in regular_business_days]
ONE_DAY_IN_SECS = 86400
OBSERVATION_TIME = 2 # days after making top gainer list
BASE_DIR = "C:\\Users\\georg\\GitHub\\money-tree\\"

# This pickle is a dictionary of date to df of premarket top gainers every day. 
PREMARKET_TOP_GAINERS_PICKLE = os.path.join(BASE_DIR, "data", "premarket_top_gainers.pkl")
PREMAREKT_TOP_GAINERS = dict()
if os.path.exists(PREMARKET_TOP_GAINERS_PICKLE):
    with open(PREMARKET_TOP_GAINERS_PICKLE, 'rb') as f:
        PREMAREKT_TOP_GAINERS = pickle.load(f)
else:
    open(PREMARKET_TOP_GAINERS_PICKLE, "x")

# This pickle is a dictionary of date to df of daily top gainers in market hours. 
DAILY_TOP_GAINERS_PICKLE = os.path.join(BASE_DIR, "data", "daily_top_gainers.pkl")
DAILY_TOP_GAINERS = dict()
if os.path.exists(DAILY_TOP_GAINERS_PICKLE):
    with open(DAILY_TOP_GAINERS_PICKLE, 'rb') as f:
        DAILY_TOP_GAINERS = pickle.load(f)
else:
    open(DAILY_TOP_GAINERS_PICKLE, "x")

job_last_ran = dt.datetime.today().date() - dt.timedelta(days=2)

# What happens to premarket gainers during the day?
def premarket_top_gainers_job():
    global date_today, business_day_index
    
    _, df = (
        Scanner.premarket_gainers
        .select('name', 'close', 'volume', 'premarket_change', 'premarket_volume', 'relative_volume_10d_calc', 'market_cap_basic', 'float_shares_outstanding')
        .where(
        Column('market_cap_basic') >= 1_000_000,
        Column('relative_volume_10d_calc') > 1.2,
        Column('premarket_change') > 5,
        Column('premarket_volume') > 1E6,
        Column('close') > 1
        )
        .get_scanner_data())


    df.volume = df.volume * 1E-6
    df.market_cap_basic = df.market_cap_basic * 1E-6
    df.premarket_volume = df.premarket_volume * 1E-6
    df.float_shares_outstanding = df.float_shares_outstanding * 1E-6
    df = df.rename(columns={
        "volume": "Volume (M)", 
        "market_cap_basic": "Market Cap (M)",
        "premarket_change": "Premarket %",
        "premarket_change_abs": "Premarket $",
        "premarket_volume": "Premarket Volume (M)",
        "float_shares_outstanding": "Float (M)"
        })

    PREMAREKT_TOP_GAINERS[date_today] = df
    with open(PREMARKET_TOP_GAINERS_PICKLE, 'wb') as f:
        pickle.dump(PREMAREKT_TOP_GAINERS, f)
    print(f"{date_today}: Found premarket top gainers", df.name.to_list())
    
    # Download 1m, 5m, 15m, 30m, 1h of all
    # Move everything to gdrive & incorporate upload here, instead of in github

# What happens to daily top gainers in a few days?
def daily_top_gainers_job():
    global date_today, business_day_index
    
    _, df = (Query()
    .select('name', 'close', 'change', 'volume', 'relative_volume_10d_calc', 'market_cap_basic', 'float_shares_outstanding')
    .where(
        Column('market_cap_basic') >= 1_000_000,
        Column('relative_volume_10d_calc') > 1.2,
        Column('change') > 5,
        Column('volume') > 1E6,
        Column('close') > 2,
        )
    .order_by('change', ascending=False)
    .get_scanner_data())

    df.volume = df.volume * 1E-6
    df.market_cap_basic = df.market_cap_basic * 1E-6
    df.float_shares_outstanding = df.float_shares_outstanding * 1E-6
    df = df.rename(columns={
        "volume": "Volume (M)", 
        "market_cap_basic": "Market Cap (M)",
        "change": "Change %",
        "float_shares_outstanding": "Float (M)"
        })
    df = df[df.ticker.str.contains("NASDAQ:") | df.ticker.str.contains("NYSE:")]
    
    DAILY_TOP_GAINERS[dt.datetime.now().date()] = df
    with open(DAILY_TOP_GAINERS_PICKLE, 'wb') as f:
        pickle.dump(DAILY_TOP_GAINERS, f)

    stocks_to_download = set()
    for day_delta in range(OBSERVATION_TIME + 1):
        lookback_date = REGULAR_BUSINESS_DAYS[business_day_index - day_delta]
        if lookback_date in DAILY_TOP_GAINERS:
            stocks_to_download.update(DAILY_TOP_GAINERS[lookback_date].name)
        else:
            print("Skipped downloading daily top gainers on", lookback_date)
    print("Downloading data for recent daily top gainers:", stocks_to_download)
    # Download 1m, 5m, 15m, 30m, 1h, 4h, 1d of all
    # Move everything to gdrive & incorporate upload here, instead of in github

if __name__ == '__main__':
    while True:
        date_today = dt.datetime.now().date()
        if date_today in REGULAR_BUSINESS_DAYS:
            business_day_index = REGULAR_BUSINESS_DAYS.index(date_today)
            if True: # dt.datetime.now().hour >= 14 and job_last_ran.date() != date_today:
                premarket_top_gainers_job()
                daily_top_gainers_job()
                job_last_ran = dt.datetime.now().date()
                print("Finished all jobs of today!")

        time.sleep(ONE_DAY_IN_SECS)