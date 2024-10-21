import os 
import pandas as pd
import datetime as dt
import numpy as np
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.join(os.environ["BASE_DIR"])

VALID_INTERVALS = {"1m","2m","5m","15m","30m","1h","4h","1d"} # "60m","90m","5d","1wk","1mo","3mo"
# NOTE: "4h" is more useful for forex & crypto. Download the data for now, but note that candle starts at 8am instead of 9:30am. 

def read_pickle_or_none(filepath):
    if os.path.exists(filepath):
        return pd.read_pickle(filepath)
    else:
        return None

def pickle_filepath(ticker, interval):
    return os.path.join(BASE_DIR, "data", f"{ticker.upper()}_{interval.lower()}.pkl")

def get_downloaded_data_or_none(ticker, interval, start=None, end=None):
    df = read_pickle_or_none(pickle_filepath(ticker, interval))
    if df is not None and len(df) > 0:
        if start is not None and end is not None:
            return df[(df['Datetime'].dt.date >= start) & (df['Datetime'].dt.date <= end)]
        elif start is not None:
            return df[(df['Datetime'].dt.date >= start)]
        elif end is not None:
            return df[(df['Datetime'].dt.date <= end)]
        else:
            return df
    return None

