import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ✅ Enable GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ✅ Load dataset
file_path = "load_data/^NSEI_merged.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# ✅ Select Features (Removed MACD to reduce noise)
features = ["Close", "Positive_News", "Negative_News", "Neutral_News", "SMA_50", "SMA_200", "RSI"]
df = df[features]

# ✅ Scale 'Close' Price Separately
price_scaler = MinMaxScaler(feature_range=(0, 1))
df["Close"] = price_scaler.fit_transform(df[["Close"]])

# ✅ Scale the Sentiment & Technical Features
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = feature_scaler.fit_transform(df)

# ✅ Create Sequences for LSTM (Longer Time-Series Window)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting Close Price
    return np.array(X), np.array(y)

SEQ_LENGTH = 120  # 🔺 Increased for better long-term trend detection
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# ✅ Split Data (80% Train, 20% Test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ✅ Model Path (Saves Model for Future Use)
model_path = "load_data/saved_lstm_model.h5"

# ✅ Check if Model Exists, Otherwise Train a New One
if os.path.exists(model_path):
    print("✅ Loading existing trained model...")
    model = load_model(model_path)
else:
    print("🚀 Training new model...")
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=False),  # 🔺 Last LSTM layer should not return sequences
        Dropout(0.4),  # 🔺 Increased Dropout for better regularization
        BatchNormalization(),

        Dense(32, activation='relu'),
        Dense(1)
    ])

    # 🔹 AdamW Optimizer for Stability
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.00005), loss='mse')
    model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

    # ✅ Save the trained model
    model.save(model_path)
    print("✅ Model saved for future use!")

# ✅ Predict & Inverse Transform
predictions = model.predict(X_test)
y_test_original = price_scaler.inverse_transform(y_test.reshape(-1, 1))  # 🔺 Fix scaling issue
predictions = price_scaler.inverse_transform(predictions)

# ✅ Calculate RMSE, MAE & Normalized RMSE
rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
mae = mean_absolute_error(y_test_original, predictions)
avg_price = np.mean(y_test_original)
normalized_rmse = rmse / avg_price

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ Normalized RMSE: {normalized_rmse:.4f} (closer to 0 is better)")
print(f"✅ MAE: {mae:.2f}")

# ✅ Plot Predictions
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_original, label="Actual Price", color="blue")
plt.plot(df.index[-len(y_test):], predictions, label="Predicted Price", linestyle="dashed", color="red")
plt.legend()
plt.title(f"LSTM Stock Price Prediction (RMSE: {rmse:.2f})")
plt.show()
