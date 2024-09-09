from tradingview_screener import *
import datetime as dt
import time
import pickle
import os
from download import *
from tqdm import tqdm

ONE_HOUR_IN_SECS = 3600
OBSERVATION_TIME = 2 # days after making top gainer list

# This pickle is a dictionary of date to df of premarket top gainers every day. 
PREMARKET_TOP_GAINERS_PICKLE = os.path.join(BASE_DIR, "data", "premarket_top_gainers.pkl")
PREMARKET_TOP_GAINERS = dict()
if os.path.exists(PREMARKET_TOP_GAINERS_PICKLE):
    with open(PREMARKET_TOP_GAINERS_PICKLE, 'rb') as f:
        PREMARKET_TOP_GAINERS = pickle.load(f)
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

    PREMARKET_TOP_GAINERS[date_today] = df
    with open(PREMARKET_TOP_GAINERS_PICKLE, 'wb') as f:
        pickle.dump(PREMARKET_TOP_GAINERS, f)
    print(f"{date_today}: Found today's premarket top gainers", df.name.to_list())
    
    # Download 1m, 5m, 15m, 30m, 1h of all
    for ticker in tqdm(df.name.to_list()):
        for interval in ["1m", "5m", "15m", "30m", "1h"]:
            get_historical_data(ticker, interval, date_today, date_today)
    print(f"{date_today}: Premarket top gainers download complete!")

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
    
    DAILY_TOP_GAINERS[date_today] = df
    with open(DAILY_TOP_GAINERS_PICKLE, 'wb') as f:
        pickle.dump(DAILY_TOP_GAINERS, f)

    stocks_to_download = set()
    for day_delta in range(OBSERVATION_TIME + 1):
        lookback_date = FULL_BUSINESS_DAYS[business_day_index - day_delta]
        if lookback_date in DAILY_TOP_GAINERS:
            stocks_to_download.update(DAILY_TOP_GAINERS[lookback_date].name)
        else:
            print("Skipped downloading daily top gainers on", lookback_date)
    print(f"{date_today}: Found recent daily top gainers:", stocks_to_download)
    
    # Download 1m, 5m, 15m, 30m, 1h, 4h, 1d of all
    for ticker in tqdm(stocks_to_download):
        for interval in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            get_historical_data(ticker, interval, date_today, date_today)
    print(f"{date_today}: Daily top gainers download complete!")

if __name__ == '__main__':
    
    test_mode = False
    test_date = dt.date(2024, 9, 10)
    
    while True:
        date_today = dt.date.today() if not test_mode else test_date
        if 1 <= date_today.month < 12 and BUSINESS_DAYS_UPDATED:
            BUSINESS_DAYS_UPDATED = False
        if date_today.month == 12 and date_today.day > 25 and not BUSINESS_DAYS_UPDATED:
            update_business_days_array()
        
        if date_today in FULL_BUSINESS_DAYS:
            business_day_index = FULL_BUSINESS_DAYS.index(date_today)
            # All jobs are meant to be run after market close. 
            if (dt.datetime.now().hour >= 21 and job_last_ran != date_today) or test_mode:
                premarket_top_gainers_job()
                daily_top_gainers_job()
                if not test_mode:
                    update_big_names_data()
                job_last_ran = dt.date.today()
                print(f"{date_today}: Finished all jobs of today!")
        
        print(f"{dt.datetime.now()}: Sleeping...")
        time.sleep(ONE_HOUR_IN_SECS)
