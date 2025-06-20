import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
print("hello")
# Load data
file_path = r"C:\Users\ayush\PycharmProjects\pythonProject\archive\sap500.csv"
df = pd.read_csv(file_path)

# Sort and clean
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df[['Close']]

# Create sliding window features
window_size = 60  # Use past 60 days
X = []
y = []

for i in range(window_size, len(df)):
    X.append(df['Close'].iloc[i - window_size:i].values)
    y.append(df['Close'].iloc[i])

X = np.array(X)
y = np.array(y)

# Train on full dataset
model = RandomForestRegressor(n_estimators=40, random_state=42)
model.fit(X, y)

# Predict next day's close using last 60 days
last_days = df['Close'].iloc[-window_size:].values.reshape(1, -1)
tomorrow_pred = model.predict(last_days)[0]
print("Predicted Close for Next Trading Day:", round(tomorrow_pred, 2))
