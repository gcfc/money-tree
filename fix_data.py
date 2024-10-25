from download import *
from tqdm import tqdm

# This pickle is a dictionary of date to df of premarket top gainers every day. 
PREMARKET_TOP_GAINERS_PICKLE = os.path.join(BASE_DIR, "data", "premarket_top_gainers.pkl")
PREMARKET_TOP_GAINERS = dict()
if os.path.exists(PREMARKET_TOP_GAINERS_PICKLE):
    with open(PREMARKET_TOP_GAINERS_PICKLE, 'rb') as f:
        PREMARKET_TOP_GAINERS = pickle.load(f)
else:
    raise FileNotFoundError("Cannot find premarket top gainers pickle file.")

# This pickle is a dictionary of date to df of daily top gainers in market hours. 
DAILY_TOP_GAINERS_PICKLE = os.path.join(BASE_DIR, "data", "daily_top_gainers.pkl")
DAILY_TOP_GAINERS = dict()
if os.path.exists(DAILY_TOP_GAINERS_PICKLE):
    with open(DAILY_TOP_GAINERS_PICKLE, 'rb') as f:
        DAILY_TOP_GAINERS = pickle.load(f)
else:
    raise FileNotFoundError("Cannot find daily top gainers pickle file.")


# 正片开始
premarket_intervals = ["1m", "5m", "15m", "30m", "1h"]
for date, premarket_df in tqdm(PREMARKET_TOP_GAINERS.items()):
    tickers = premarket_df.name
    for ticker in tickers:
        for interval in premarket_intervals:
            download_and_save(ticker, interval, date, date, allow_low_quality=True)
            # to_fix = False
            # if not os.path.exists(pickle_filepath(ticker, interval)):
            #     to_fix = True
            # else:
            #     data = get_downloaded_data_or_none(ticker, interval, date, date)
            #     if data is None or len(data) == 0:
            #         to_fix = True
            # if to_fix:
            #     download_and_save(ticker, interval, date, date)

for date, daily_df in tqdm(DAILY_TOP_GAINERS.items()):
    tickers = daily_df.name
    for ticker in tickers:
        for interval in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            bday_idx = FULL_BUSINESS_DAYS.index(date)
            for idx in [bday_idx, bday_idx + 1, bday_idx + 2]:
                if FULL_BUSINESS_DAYS[idx] < dt.date.today():
                    download_and_save(ticker, interval, FULL_BUSINESS_DAYS[idx], FULL_BUSINESS_DAYS[idx], allow_low_quality=True)
