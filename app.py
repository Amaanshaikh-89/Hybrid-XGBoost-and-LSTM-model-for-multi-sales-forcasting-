import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from xgboost import XGBRegressor

# ------------------------------- CREATE IMAGES FOLDER -------------------------------
if not os.path.exists("images"):
    os.makedirs("images")

# ------------------------------- LOAD DATA -------------------------------
df = pd.read_csv("data/walmart.csv")

# ------------------------------- PREPROCESSING -------------------------------
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values(by=['Store', 'Date'])

df.rename(columns={
    'Store': 'store_id',
    'Date': 'date',
    'Weekly_Sales': 'sales',
    'Holiday_Flag': 'holiday'
}, inplace=True)

# ------------------------------- NORMALIZE STORE ID -------------------------------
df['store_id'] = df['store_id'] / df['store_id'].max()

# ------------------------------- FEATURE ENGINEERING -------------------------------
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# ------------------------------- LAG FEATURES -------------------------------
df['lag_1'] = df.groupby('store_id')['sales'].shift(1)
df['lag_7'] = df.groupby('store_id')['sales'].shift(7)

df = df.dropna()

# ------------------------------- SCALE FEATURES -------------------------------
features_lstm = ['store_id', 'sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

scaler = MinMaxScaler()
df[features_lstm] = scaler.fit_transform(df[features_lstm])

# ------------------------------- CREATE SEQUENCES -------------------------------
def create_sequences_multistore(df, window_size, features):
    X, y = [], []

    for store in df['store_id'].unique():
        store_data = df[df['store_id'] == store].sort_values('date')
        data = store_data[features].values

        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size][1])

    return np.array(X), np.array(y)

window_size = 10
X, y = create_sequences_multistore(df, window_size, features_lstm)

# ------------------------------- TRAIN TEST SPLIT -------------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------------- LSTM MODEL -------------------------------
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=60, batch_size=32, shuffle=True, verbose=1)

# ------------------------------- PREDICTIONS -------------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ------------------------------- INVERSE SCALING -------------------------------
def inverse_scale(pred, scaler):
    temp = np.zeros((len(pred), len(features_lstm)))
    temp[:, 1] = pred.flatten()
    return scaler.inverse_transform(temp)[:, 1]

train_pred = inverse_scale(train_pred, scaler)
test_pred = inverse_scale(test_pred, scaler)

y_train_actual = inverse_scale(y_train.reshape(-1,1), scaler)
y_test_actual = inverse_scale(y_test.reshape(-1,1), scaler)

# ------------------------------- XGBOOST DATA -------------------------------
df_xgb = df.copy()
df_xgb = df_xgb.groupby('store_id').apply(lambda x: x.iloc[window_size:]).reset_index(drop=True)

features_xgb = ['store_id', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                'day', 'month', 'day_of_week', 'lag_1', 'lag_7']

X_xgb = df_xgb[features_xgb]

X_xgb_train = X_xgb[:split]
X_xgb_test = X_xgb[split:]

train_residuals = y_train_actual - train_pred
test_residuals = y_test_actual - test_pred

# ------------------------------- XGBOOST MODEL -------------------------------
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
xgb_model.fit(X_xgb_train, train_residuals)

# ------------------------------- HYBRID PREDICTION -------------------------------
residual_pred = xgb_model.predict(X_xgb_test)
final_predictions = test_pred + residual_pred

# ------------------------------- METRICS -------------------------------
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, test_pred))
rmse_hybrid = np.sqrt(mean_squared_error(y_test_actual, final_predictions))

mae_lstm = mean_absolute_error(y_test_actual, test_pred)
mae_hybrid = mean_absolute_error(y_test_actual, final_predictions)

print("\nModel Performance:")
print(f"LSTM RMSE: {rmse_lstm:.2f}")
print(f"Hybrid RMSE: {rmse_hybrid:.2f}")
print(f"LSTM MAE: {mae_lstm:.2f}")
print(f"Hybrid MAE: {mae_hybrid:.2f}")

# ------------------------------- GRAPH 1 -------------------------------
plt.figure()
plt.plot(y_test_actual[:100], label='Actual')
plt.plot(test_pred[:100], label='LSTM')
plt.plot(final_predictions[:100], label='Hybrid')
plt.legend()
plt.title("Actual vs LSTM vs Hybrid")
plt.savefig("images/graph1_main.png")
plt.close()

# ------------------------------- GRAPH 2 -------------------------------
residuals = y_test_actual - test_pred
plt.figure()
plt.plot(residuals[:100])
plt.title("Residual Errors (LSTM)")
plt.savefig("images/graph2_residual.png")
plt.close()

# ------------------------------- GRAPH 3 -------------------------------
plt.figure()
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.savefig("images/graph3_loss.png")
plt.close()

# ------------------------------- GRAPH 4 -------------------------------
models = ['LSTM RMSE', 'Hybrid RMSE']
values = [rmse_lstm, rmse_hybrid]

plt.figure()
plt.bar(models, values)
plt.title("Model Comparison (RMSE)")
plt.savefig("images/graph4_comparison.png")
plt.close()

# ------------------------------- GRAPH 5 -------------------------------
plt.figure()
plt.plot(y_test_actual[:50], label='Actual')
plt.plot(final_predictions[:50], label='Hybrid')
plt.legend()
plt.title("Zoomed Prediction")
plt.savefig("images/graph5_zoom.png")
plt.close()

print("\nAll graphs saved successfully in 'images/' folder!")