#%%
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from calendar import monthrange
from stock_indicators import indicators
from stock_indicators.indicators.common.quote import Quote
from tqdm import tqdm

TICKER = "SPY"
TICKER = TICKER.upper()
NUM_YEARS = 50 # SPY only began in 1993
start_date = dt.date(dt.date.today().year - NUM_YEARS, 1, 1)
end_date = dt.date(dt.date.today().year, 1, 1)
print("Downloading...")
data = yf.download(TICKER, str(start_date), str(end_date))
start_date = max(start_date, data.index[0].to_pydatetime().date())
NUM_YEARS = (end_date - start_date).days / 365.25
print(f"{TICKER}:")

quotes = []
for i in tqdm(range(len(data))):
    quotes.append(Quote(date=data.index.to_pydatetime()[i], open=data['Open'][i], high=data['High'][i], low=data['Low'][i], close=data['Adj Close'][i], volume=data['Volume'][i]))

#%% Indicators
rsi_results = indicators.get_rsi(quotes)
rsi = {result.date.date() : result.rsi for result in rsi_results}
macd_results = indicators.get_macd(quotes)
macd = {result.date.date() : result for result in macd_results}
ema_9_results = indicators.get_ema(quotes, lookback_periods=9)
ema_9 = {result.date.date() : result.ema for result in ema_9_results}
sma_150 = indicators.get_sma(quotes, lookback_periods=200)
sma_slope = np.diff(np.array([sma_150[i].sma - sma_150[i-1].sma for i in range(1, len(sma_150)) if sma_150[i-1].sma is not None]), 1)
dates = list(map(lambda x: x.to_pydatetime().date(), data.index))
prices_dict = {d : p for d, p in zip(dates, data['Adj Close'].tolist())}


#%% Trading 
prev_date, curr_date = None, start_date
curr_shares = 0
projection_trading = 0
x_trading, y_trading, in_trade_history, bought_history = [], [], [], []
in_trade = False
money_in, daily_gains, curr_gains = 0, 0, 0
max_shares, max_money_in = 0, 0

while curr_date < end_date:
    if curr_date in prices_dict.keys() and rsi[curr_date] and ema_9[curr_date] and macd[curr_date].macd and macd[curr_date].signal:
        if prev_date is not None and in_trade: 
            daily_gains += curr_shares * (prices_dict[curr_date] - prices_dict[prev_date])
            curr_gains += daily_gains
            projection_trading += daily_gains
        
        if (rsi[curr_date] < 40) \
            and (macd[curr_date].macd > macd[curr_date].signal) and (macd[curr_date].macd < 0) and (macd[curr_date].signal < 0):
            in_trade = True
            curr_shares += 3
            money_in += curr_shares * prices_dict[curr_date]
            max_shares = max(curr_shares, max_shares)
            max_money_in = max(money_in, max_money_in)
            
        elif (in_trade and macd[curr_date].macd < macd[curr_date].signal):
            print(curr_gains)
            in_trade = False
            curr_shares = 0
            money_in = 0
            curr_gains = 0
        
        prev_date = curr_date
        x_trading.append(curr_date)
        y_trading.append(projection_trading)
        in_trade_history.append(1 if in_trade else np.nan)

    # Tomorrow, loop until first market open date
    curr_date += dt.timedelta(days=1)

print(f"Total earning: \t ${round(projection_trading, 2):,}")
print(f"Max shares: {max_shares}")
print(f"Max money in: ${round(max_money_in, 2):,}")


#%% Plotting
import numpy as np
total_plots = 4
start_ind = -500
end_ind = -1
plt.style.use("dark_background")
fig = plt.gcf()
fig.set_size_inches(10, 20)
fig.suptitle(f"Investment Strategies on {TICKER}")

plt.subplot(total_plots, 1, 1)
plt.plot(data.index[start_ind:end_ind], data['Adj Close'][start_ind:end_ind])
plt.plot(data.index[start_ind:end_ind], (np.array(data['Adj Close'][-len(in_trade_history):]) * np.array(in_trade_history))[start_ind:end_ind])
plt.grid(visible=True, which="both", axis="both")
plt.ylabel("Ticker Price")
plt.title(f"{TICKER} Daily Closing Price")

plt.subplot(total_plots, 1, 2)
plt.plot(dates[start_ind:end_ind], [result.macd for result in macd_results][start_ind:end_ind])
plt.plot(dates[start_ind:end_ind], [result.signal for result in macd_results][start_ind:end_ind])
plt.plot(dates[start_ind:end_ind], (np.array([result.macd for result in macd_results][-len(in_trade_history):]) * np.array(in_trade_history))[start_ind:end_ind])
plt.grid(visible=True, which="both", axis="both")
plt.title("MACD")

plt.subplot(total_plots, 1, 3)
plt.plot(list(rsi.keys())[start_ind:end_ind], np.array(list(rsi.values()))[start_ind:end_ind])
plt.plot(list(rsi.keys())[start_ind:end_ind], (np.array(list(rsi.values())[-len(in_trade_history):]) * np.array(in_trade_history))[start_ind:end_ind])
plt.grid(visible=True, which="both", axis="both")
plt.title("RSI")

plt.subplot(total_plots, 1, 4)
plt.plot(x_trading[start_ind:end_ind], y_trading[start_ind:end_ind])
plt.plot(x_trading[start_ind:end_ind], (np.array(y_trading) * np.array(in_trade_history))[start_ind:end_ind])
plt.grid(visible=True, which="both", axis="both")
plt.title("P&L")


plt.tight_layout()
plt.show()


# %%
# aapl = yf.Ticker("AAPL")
# data = aapl.history(period="60d", interval="30m")
# print(data["Close"])
# 1h - 730d
# 30m, 15m, 5m, 2m - 60d
# 1m - 7d