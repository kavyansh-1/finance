"""
DAY 3 — Trading Bot Project
Goal: Add risk management to the strategy:
  1. Stop-loss  -> exit a trade automatically if it drops too far
  2. Position sizing -> never risk your whole account on one trade
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TICKER="AAPL"
PERIOD="1y"
INTERVAL="1d"

import datetime
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)
data = yf.download(TICKER, start=start_date, end=end_date, interval="1d")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data["SMA_short"]=data["Close"].rolling(window=20).mean()
data["SMA_long"]=data["Close"].rolling(window=50).mean()

data["Signal"]=0
data.loc[data["SMA_short"]>data["SMA_long"],"Signal"]=1

print(data["Signal"].value_counts())
print("First few signal values:", data["Signal"].head(60).tolist())

STARTING_CASH=10_000
STOP_LOSS_PCT=0.05
RISK_PER_TRADE=0.02

cash=STARTING_CASH
shares_held=0
entry_price=None
equity_curve=[]
trade_log=[]

for i in range(1, len(data)):
    price_today = data["Close"].iloc[i]
    signal_today = data["Signal"].iloc[i]
    signal_yesterday = data["Signal"].iloc[i - 1]

    if signal_today == 1 and signal_yesterday == 0:
        print(f"Flip detected on {data.index[i].date()} | shares_held={shares_held} | price={price_today} | cash={cash}")

    #---- Check stop loss ----
    if shares_held > 0 and entry_price is not None:
        loss_pct = (price_today - entry_price) / entry_price
        if loss_pct <= -STOP_LOSS_PCT:
            cash += shares_held * price_today
            trade_log.append((data.index[i], "STOP-LOSS SELL", price_today, loss_pct))
            shares_held = 0
            entry_price = None

    # --- Entry: crossover just turned bullish, and we're not already in ---
    # NOTE: dedented -- this is now OUTSIDE the stop-loss "if" block above
    if signal_today == 1 and signal_yesterday == 0 and shares_held == 0:
        risk_amount = cash * RISK_PER_TRADE
        dollars_to_risk_per_share = price_today * STOP_LOSS_PCT
        shares_to_buy = int(risk_amount / dollars_to_risk_per_share)

        cost = shares_to_buy * price_today
        if shares_to_buy > 0 and cash >= cost:
            cash -= cost
            shares_held = shares_to_buy
            entry_price = price_today
            trade_log.append((data.index[i], "BUY", price_today, shares_to_buy))  # <-- fixed parens

    # --- Exit: crossover turned bearish ---
    elif signal_today == 0 and signal_yesterday == 1 and shares_held > 0:
        cash += shares_held * price_today
        trade_log.append((data.index[i], "SIGNAL SELL", price_today, shares_held))  # <-- fixed parens
        shares_held = 0
        entry_price = None

    #--- Track total account value each day (cash + any oprn position) ---
    total_value = cash + shares_held * price_today
    equity_curve.append(total_value)

data=data.iloc[1:].copy()  
data["Equity"] = equity_curve


#Compare to buy and hold
data["BuyHold_Equity"]= STARTING_CASH*(data["Close"]/data["Close"].iloc[0])

final_equity=data["Equity"].iloc[-1]
final_buyhold=data["BuyHold_Equity"].iloc[-1]

print(f"--- Results for {TICKER} (starting with ${STARTING_CASH:,}) ---")
print(f"Strategy w/ risk mgmt final value: ${final_equity:,.2f}  ({(final_equity/STARTING_CASH - 1)*100:.1f}%)")
print(f"Buy & Hold final value:            ${final_buyhold:,.2f}  ({(final_buyhold/STARTING_CASH - 1)*100:.1f}%)")

print(f"\n--- Trade Log ({len(trade_log)} events) ---")
for date, action, price, extra in trade_log:
    print(f"{date.date()}  {action:<14}  ${price:.2f}  {extra}")


#PLot
plt.figure(figsize=(14,7))
plt.plot(data["Equity"],label="Startegy w/ Stop-Loss + Position Sizing",linewidth=2)
plt.plot(data["BuyHold_Equity"],label="Buy and Hold",linewidth=2,alpha=0.7)
plt.title(f"{TICKER} — Strategy with Risk Management vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Account Value ($)")
plt.legend()
plt.tight_layout()
plt.savefig("DAY3/risk_managed_strategy.png")
print("\nChart saved as risk_managed_strategy.png")