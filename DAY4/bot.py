"""
DAY 4 — Trading Bot Project
Goal: Use XGBoost to predict next-day price DIRECTION (up/down),
      using engineered features instead of raw price.
 
IMPORTANT: We evaluate this HONESTLY -- time-based split (never shuffle
time series data), and we compare against a "dumb baseline" (always
predict "up") because markets go up more often than down historically,
so a model needs to beat THAT, not just 50%.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt

TICKER="AAPL"
data=yf.download(TICKER,period="5y",interval="1d")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df=pd.DataFrame(index=data.index)
close=data["Close"]
volume=data["Volume"]

df["return_1d"]=close.pct_change(1)
df["return_5d"]=close.pct_change(5)
df["return_10d"]=close.pct_change(10)

df["sma_10d"]=close.rolling(window=10).mean()
df["sma_50d"]=close.rolling(window=50).mean()
df["price_vs_sma10d"]=(close-df["sma_10d"])/df["sma_10d"]
df["price_vs_sma50d"]=(close-df["sma_50d"])/df["sma_50d"]

df["volatility_10d"]=close.pct_change().rolling(10).std()

df["volume_change"]=volume.pct_change(5)

#RSI
delta=close.diff()
gain=delta.where(delta > 0.0,0).rolling(14).mean()
loss=-delta.where(delta < 0.0,0).rolling(14).mean()
rs=gain/loss
df["rsi"]=100 - (100/(1+rs))



# Define the TARGET (what we're predicting)

# 1 if tomorrow's close is higher than todays close, else 0
df["target"]=(close.shift(-1) > close).astype(int)

df=df.dropna()

feature_cols = ["return_1d", "return_5d", "return_10d", "price_vs_sma10d",
                 "price_vs_sma50d", "volatility_10d", "volume_change", "rsi"]

X=df[feature_cols]
Y=df["target"]

split_data=int(len(df)*0.8)
X_train,X_test=X.iloc[:split_data],X.iloc[split_data:]
Y_train,Y_test=Y.iloc[:split_data],Y.iloc[split_data:]

print(f"Training on {len(X_train)} days, testing on {len(X_test)} days")

model=XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train,Y_train)


preds=model.predict(X_test)
accuracy=accuracy_score(Y_test,preds)

baseline_accuracy=(Y_test == 1).mean()

print(f"\n--- Results ---")
print(f"Model accuracy:            {accuracy*100:.1f}%")
print(f"Baseline ('always up'):    {baseline_accuracy*100:.1f}%")
print(f"Model beats baseline by:   {(accuracy - baseline_accuracy)*100:+.1f} percentage points")
 
print("\n--- Detailed report ---")
print(classification_report(Y_test, preds, target_names=["Down", "Up"]))

importances=pd.Series(model.feature_importances_,index=feature_cols).sort_values(ascending=False)

print("\n--- Feature importance ---")
print(importances)
 
plt.figure(figsize=(10, 6))
importances.plot(kind="barh")
plt.title(f"{TICKER} — XGBoost Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("DAY4/feature_importance.png")
print("\nChart saved as feature_importance.png")

