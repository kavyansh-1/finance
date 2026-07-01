"""
DAY 1 — Trading Bot Project
Goal: Pull historical stock data and calculate a moving average crossover signal.
No trading yet — just learning to handle price data.
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

TICKER="AAPL"
PERIOD="1y"
INTERVAL="1d"

data=yf.download(TICKER,period=PERIOD,interval=INTERVAL)

print(f"Downloaded {len(data)} row for {TICKER}")
print(data.tail())

#Calculate moving average
data["SMA_short"]=data["Close"].rolling(window=20).mean()

data["SMA_long"]=data["Close"].rolling(window=50).mean()

#Generate a signal
data["signal"]=0
data.loc[data["SMA_short"] > data["SMA_long"] , "signal"] = 1

# "Position" marks the exact day the crossover happens (the trade trigger)
data["Position"]=data["signal"].diff()
# Position ==  1  -> crossover UP   (this is a BUY day)
# Position == -1  -> crossover DOWN (this is a SELL day)

#Show the trade signals
buy_signals=data[data["Position"]==1]
sell_signals=data[data["Position"]==-1]

print(f"\nNumber of BUY signals: {len(buy_signals)}")
print(f"Number of SELL signals: {len(sell_signals)}")
 

# ----------------------------
# STEP 5: Plot it so you can SEE what's happening
# ----------------------------
plt.figure(figsize=(14, 7))
plt.plot(data["Close"], label="Close Price", alpha=0.5)
plt.plot(data["SMA_short"], label="SMA 20", alpha=0.9)
plt.plot(data["SMA_long"], label="SMA 50", alpha=0.9)
 
plt.scatter(buy_signals.index, data.loc[buy_signals.index, "Close"],
            marker="^", color="green", s=120, label="BUY", zorder=5)
plt.scatter(sell_signals.index, data.loc[sell_signals.index, "Close"],
            marker="v", color="red", s=120, label="SELL", zorder=5)
 
plt.title(f"{TICKER} — SMA Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("sma_crossover.png")
print("\nChart saved as sma_crossover.png")