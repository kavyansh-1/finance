"""
DAY 2 — Trading Bot Project
Goal: Turn buy/sell signals into actual $ returns, and compare
      the strategy vs simply buying and holding the stock.
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

TICKER="AAPL"
PERIOD="1y"
INTERVAL="1d"

data=yf.download(TICKER, period=PERIOD, interval=INTERVAL)

data["SMA_short"]=data["Close"].rolling(window=20).mean()
data["SMA_long"]=data["Close"].rolling(window=50).mean()

data["Signal"]=0
data.loc[data["SMA_short"]>data["SMA_long"], "Signal"]=1
data["Position"]=data["Signal"].diff()

# Daily % change of the actual stock price
data["Daily_Return"]=data["Close"].pct_change()

# Strategy return = daily stock return, but ONLY on days we were "in" a position.
# We shift Signal by 1 day because you can only act on a signal the day AFTER it appears
# (you see yesterday's close, so you trade at today's open/close — never the same bar).
data["Strategy_Return"]=data["Daily_Return"]*data["Signal"].shift(1)

# "If you started with $1, how much would you have now?"
data["BuyHold_Growth"]=(1+data["Daily_Return"]).cumprod()
data["Strategy_Growth"]=(1+data["Strategy_Return"]).cumprod()

final_buyhold=data["BuyHold_Growth"].iloc[-1]
final_strategy=data["Strategy_Growth"].iloc[-1]

print(f"--- Results for {TICKER} ---")
print(f"Buy & Hold final value (from $1):   ${final_buyhold:.2f}  ({(final_buyhold-1)*100:.1f}%)")
print(f"Strategy final value (from $1):     ${final_strategy:.2f}  ({(final_strategy-1)*100:.1f}%)")


# Max drawdown = the worst peak-to-trough drop you'd have experienced.
# This matters MORE than total return in real trading -- it tells you
# how much pain you'd have sat through.
def max_drawdown(growth_series):
    running_max=growth_series.cummax()
    drawdown=(growth_series-running_max)/running_max
    return drawdown.min()

print(f"\nBuy & Hold max drawdown: {max_drawdown(data['BuyHold_Growth'])*100:.1f}%")
print(f"Strategy max drawdown:   {max_drawdown(data['Strategy_Growth'])*100:.1f}%")
 
# Win rate: of the trades that closed, how many were profitable?
trades = data[data["Position"] != 0].copy()
print(f"\nNumber of buy/sell signals: {len(trades)}")
 

plt.figure(figsize=(14, 7))
plt.plot(data["BuyHold_Growth"], label="Buy & Hold", linewidth=2)
plt.plot(data["Strategy_Growth"], label="SMA Crossover Strategy", linewidth=2)
plt.title(f"{TICKER} — Strategy vs Buy & Hold (growth of $1)")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.tight_layout()
plt.savefig("strategy_vs_buyhold.png")
print("\nChart saved as strategy_vs_buyhold.png")