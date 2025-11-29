# TCS Stock Prediction: Cognitive Decision-Making Framework

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import warnings
warnings.filterwarnings('ignore')

# ===== 1. Data Loading & Preparation =====
stocks = pd.read_csv(r'C:\Users\hanis\Downloads\intern\TCS-stock_forecast\data\TCS_stock_history_.csv')
stocks['Date'] = pd.to_datetime(stocks['Date'])
actions = pd.read_csv(r'C:\Users\hanis\Downloads\intern\TCS-stock_forecast\data\TCS_stock_action_.csv')
actions['Date'] = pd.to_datetime(actions['Date'])

stocks = stocks.merge(actions.rename(columns={'Dividends':'Dividend','Stock Splits':'Split'}), on='Date', how='left').fillna(0)

split_factor = stocks['Split'].replace(0,1).cumprod()
stocks['Adjusted Close'] = stocks['Close'] / split_factor
stocks['Adj Close Return'] = stocks['Adjusted Close'].pct_change()
stocks['Total Return'] = stocks['Adj Close Return'] + stocks['Dividend'] / stocks['Adjusted Close'].shift(1)

# ===== 2. Baseline EDA for Stakeholders =====
plt.figure(figsize=(10,5))
plt.plot(stocks['Date'], stocks['Close'], label='Close Price')
plt.title('TCS Close Price Over Time')
plt.xlabel('Date'); plt.ylabel('Close Price')
plt.tight_layout(); plt.savefig('eda_close_time_series.png'); plt.close()

plt.figure(figsize=(12,6))
plt.plot(stocks['Date'], stocks['Volume'], label='Volume', color='blue')
plt.plot(stocks['Date'], stocks['Dividend'], label='Dividends', color='green')
plt.plot(stocks['Date'], stocks['Split'], label='Splits', color='red')
plt.title('Volume, Dividends, and Stock Splits Over Time')
plt.xlabel('Date'); plt.ylabel('Value'); plt.legend()
plt.tight_layout(); plt.savefig('eda_vol_div_split.png'); plt.close()

plt.figure(figsize=(8,5))
plt.scatter(stocks['Volume'], stocks['Close'], alpha=0.5)
plt.title('Close Price vs. Volume')
plt.xlabel('Volume'); plt.ylabel('Close Price')
plt.tight_layout(); plt.savefig('eda_close_vs_volume.png'); plt.close()

plt.figure(figsize=(8,5))
plt.scatter(stocks['Dividend'], stocks['Close'], alpha=0.5, color='green')
plt.title('Dividends vs. Close Price')
plt.xlabel('Dividends'); plt.ylabel('Close Price')
plt.tight_layout(); plt.savefig('eda_div_vs_close.png'); plt.close()

plt.figure(figsize=(8,5))
plt.scatter(stocks['Split'], stocks['Close'], alpha=0.5, color='red')
plt.title('Stock Splits vs. Close Price')
plt.xlabel('Stock Splits'); plt.ylabel('Close Price')
plt.tight_layout(); plt.savefig('eda_split_vs_close.png'); plt.close()

stocks['MA30'] = stocks['Close'].rolling(30).mean()
plt.figure(figsize=(12,5))
plt.plot(stocks['Date'], stocks['Close'], label='Close Price')
plt.plot(stocks['Date'], stocks['MA30'], label='30-Day Moving Avg', color='orange')
plt.title('Close Price and 30-Day Moving Average')
plt.xlabel('Date'); plt.ylabel('Price'); plt.legend()
plt.tight_layout(); plt.savefig('eda_ma30.png'); plt.close()

# ===== 3. Advanced Behavioral & Feature Engineering =====
stocks['Up_Streak'] = (stocks['Close'] > stocks['Close'].shift(1)).astype(int)
stocks['Herd_Streak'] = stocks['Up_Streak'].rolling(3).sum() == 3
stocks['Herd_Streak'] = stocks['Herd_Streak'].astype(int)
stocks['MA20'] = stocks['Close'].rolling(20).mean()
stocks['Dispossession'] = (stocks['Close'] < stocks['MA20']).astype(int)
stocks['BaseRate_20'] = stocks['Total Return'].rolling(20).mean()

feat_cols = ['Open','High','Low','Close','Adjusted Close','Total Return','Dividend','Split','Volume','MA30','Herd_Streak','Dispossession','BaseRate_20']
corr_matrix = stocks[feat_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout(); plt.savefig('eda_feature_heatmap.png'); plt.close()

plt.figure(figsize=(8,4))
sns.histplot(stocks['Total Return'].dropna(), bins=70, color='steelblue', alpha=0.7)
plt.title('Distribution of Daily Total Returns')
plt.xlabel('Total Return (%)')
plt.tight_layout(); plt.savefig('eda_return_hist.png'); plt.close()

plt.figure(figsize=(14,6))
plt.plot(stocks['Date'], stocks['Total Return'], label='Total Return (w/ Dividends)', zorder=1)
plt.scatter(stocks['Date'][stocks['Herd_Streak']==1], stocks['Total Return'][stocks['Herd_Streak']==1], color='red', label='Herding (3+ up)', alpha=0.7, zorder=2)
plt.title('Total Return & Herding Events')
plt.xlabel('Date'); plt.ylabel('Total Return'); plt.legend()
plt.tight_layout(); plt.savefig('eda_return_herding.png'); plt.close()

plt.figure(figsize=(10,4))
plt.plot(stocks['Date'], stocks['BaseRate_20'], color='purple', label='20-Day Base Rate')
plt.title('Rolling 20-Day Mean of Total Return')
plt.xlabel('Date'); plt.ylabel('Base Rate (Mean Return)'); plt.legend()
plt.tight_layout(); plt.savefig('eda_base_rate.png'); plt.close()

# ===== 4. LSTM Predictive Modeling & Actual vs Predicted Plot =====
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# Use Adjusted Close for prediction
prices = stocks[['Adjusted Close']].dropna().values
scaler = MinMaxScaler(feature_range=(0,1))
prices_scaled = scaler.fit_transform(prices)

lookback = 30
X, y = [], []
for i in range(lookback, len(prices_scaled)):
    X.append(prices_scaled[i-lookback:i, 0])
    y.append(prices_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    LSTM(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

predicted_scaled = model.predict(X_test).flatten()
predicted = scaler.inverse_transform(predicted_scaled.reshape(-1,1)).flatten()
actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

plt.figure(figsize=(12,6))
plt.plot(actual, label='Actual Adjusted Close')
plt.plot(predicted, label='Predicted Adjusted Close')
plt.title('Actual vs Predicted Adjusted Close Price - LSTM')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_lstm.png')
plt.close()

# ===== 5. Summary =====
cum_return = (stocks['Adjusted Close'].iloc[-1] / stocks['Adjusted Close'].iloc[0]) - 1
summary = (
    f"TCS Stock Performance {stocks['Date'].iloc[0].date()} to {stocks['Date'].iloc[-1].date()}\n"
    f"Total Cumulative Return: {cum_return:.2%}\n"
    f"Average Daily Return: {stocks['Total Return'].mean():.2%}\n"
    f"Volatility (std dev): {stocks['Total Return'].std():.2%}\n"
    f"Best Day: {stocks['Total Return'].max():.2%} | Worst Day: {stocks['Total Return'].min():.2%}\n"
    f"Herding Streak Days (3+ up): {stocks['Herd_Streak'].sum()} | Dispossession Days (below MA20): {stocks['Dispossession'].sum()}\n"
)
print(summary)



