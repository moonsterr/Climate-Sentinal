import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow logging to only show errors
tf.get_logger().setLevel('ERROR')

# -------------------------
# User Settings
# -------------------------
# List of all data files to include
CSV_FILES = [
    "aqidaily2020.csv",
    "aqidaily2021.csv",
    "aqidaily2022.csv",
    "aqidaily2023.csv",
    "aqidaily2024.csv",
    "aqidaily2025.csv",
]
LOOKBACK = 30
EPOCHS = 60
BATCH_SIZE = 32
TEST_SPLIT_RATIO = 0.2

# Column names must be consistent across all files
DATE_COL = "Date"
AQI_COL = "Overall AQI Value"

# -------------------------
# Load, Combine, and Preprocess Data
# -------------------------
print("Loading and Combining Data from Multiple Years...")
all_data = []

for file_path in CSV_FILES:
    if not os.path.exists(file_path):
        print(f"Warning: File not found at: {file_path}. Skipping.")
        continue
    
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Validate columns
    if AQI_COL not in df.columns or DATE_COL not in df.columns:
        print(f"Warning: Columns '{DATE_COL}' or '{AQI_COL}' not found in {file_path}. Skipping.")
        continue
    
    # Select only the necessary columns and append
    all_data.append(df[[DATE_COL, AQI_COL]])

if not all_data:
    raise FileNotFoundError("No valid CSV files were loaded. Check file names and paths.")

# Concatenate all DataFrames into one large DataFrame
df_combined = pd.concat(all_data, ignore_index=True)

# Convert date column and set as index
df_combined[DATE_COL] = pd.to_datetime(df_combined[DATE_COL], errors='coerce')
df_combined = df_combined.set_index(DATE_COL).sort_index()

# Convert AQI column to numeric, coercing errors to NaN
series = df_combined[AQI_COL].apply(pd.to_numeric, errors='coerce')

# Daily resampling (ensuring continuous daily data) + interpolation (filling missing values)
# This creates the final, smooth time series used for modeling.
series = series.resample("D").mean()
series = series.interpolate(limit_direction="both")

print(f"Total continuous data points for training: {len(series)}")

# -------------------------
# Scaling
# -------------------------
values = series.values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# -------------------------
# Sequence Builder
# -------------------------
def create_sequences(data, lookback=LOOKBACK):
    """Creates input (X) and output (y) sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, 0])
        y.append(data[i+lookback, 0])
    X = np.array(X)
    y = np.array(y)
    # Reshape X for LSTM input: (samples, timesteps, features)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

X, y = create_sequences(scaled, LOOKBACK)

# -------------------------
# Train-Test Split
# -------------------------
# The entire dataset (2022-2025) is now split into train/test sets
split_idx = int(len(X) * (1 - TEST_SPLIT_RATIO))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")

# -------------------------
# Build & Train LSTM Model
# -------------------------
print("\nBuilding and Training LSTM Model...")
# LSTMs are specialized recurrent neural networks ideal for sequence data like time series.
# [Image of LSTM network structure]
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1), name='lstm_1'),
    Dropout(0.2, name='dropout_1'),
    LSTM(32, name='lstm_2'),
    Dropout(0.2, name='dropout_2'),
    Dense(1, name='output')
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# -------------------------
# Plot Training History
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History (Loss vs. Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (Loss)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

# -------------------------
# Predictions and Inverse Scaling
# -------------------------
y_pred = model.predict(X_test, verbose=0)

# Reshape for inverse_transform
y_test_reshaped = y_test.reshape(-1, 1)
y_pred_reshaped = y_pred.reshape(-1, 1)

y_test_inv = scaler.inverse_transform(y_test_reshaped)
y_pred_inv = scaler.inverse_transform(y_pred_reshaped)

# -------------------------
# Robust NaN and Infinite Value Handling (Cleaning the arrays for metrics)
# -------------------------
y_test_flat = y_test_inv.flatten()
y_pred_flat = y_pred_inv.flatten()
mask = np.isfinite(y_test_flat) & np.isfinite(y_pred_flat)
y_test_clean = y_test_flat[mask]
y_pred_clean = y_pred_flat[mask]

# -------------------------
# Metrics Calculation and Test Set Plot
# -------------------------
if len(y_test_clean) > 0:
    mae = mean_absolute_error(y_test_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
    r2 = r2_score(y_test_clean, y_pred_clean)

    print("\n--- Evaluation Metrics (Test Set) ---")
    print(f"Valid Samples = {len(y_test_clean)}")
    print(f"MAE  = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R²   = {r2:.3f}")

    # Plot Actual vs Predicted (Test Set)
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_clean, label="Actual AQI", color='blue')
    plt.plot(y_pred_clean, label="Predicted AQI", color='red', linestyle='--')
    plt.title('Actual vs Predicted AQI (Test Set)')
    plt.xlabel('Test Sample Index (Days)')
    plt.ylabel('AQI Value')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
else:
    print("\n--- Evaluation Metrics ---")
    print("WARNING: Test set is empty after cleaning NaN/Inf values. Cannot calculate metrics or plot.")


# -------------------------
# Future Forecast Function
# -------------------------
def forecast_forward(model, last_sequence, steps):
    """Performs multi-step forecasting by iteratively predicting and updating the input sequence."""
    seq = last_sequence.copy()
    output = []
    for _ in range(steps):
        # Predict the next step
        pred = model.predict(seq.reshape(1, LOOKBACK, 1), verbose=0)[0][0]
        output.append(pred)
        # Shift the sequence and append the new prediction
        seq = np.append(seq[1:], pred)
    return np.array(output)

# Get the last 'LOOKBACK' (30) scaled values from the *entire* dataset for the starting point
last_seq = scaled[-LOOKBACK:, 0]

# Generate forecasts
print("\nGenerating Future Forecasts (7, 30, and 365 days)...")
forecast_7 = forecast_forward(model, last_seq, 7)
forecast_30 = forecast_forward(model, last_seq, 30)
forecast_365 = forecast_forward(model, last_seq, 365)

# Inverse-transform forecasts
forecast_7 = scaler.inverse_transform(forecast_7.reshape(-1, 1)).flatten()
forecast_30 = scaler.inverse_transform(forecast_30.reshape(-1, 1)).flatten()
forecast_365 = scaler.inverse_transform(forecast_365.reshape(-1, 1)).flatten()

# Create dates for the forecast files and plots
last_date = series.index[-1]
dates_7 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
dates_30 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
dates_365 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)

# -------------------------
# Plot Forecasts
# -------------------------

# 7-Day Plot
plt.figure(figsize=(10, 5))
plt.plot(dates_7, forecast_7, marker='o', linestyle='-', color='red')
plt.title(f"7-Day AQI Forecast (Starting {dates_7[0].strftime('%Y-%m-%d')})")
plt.xlabel("Date")
plt.ylabel("Predicted AQI Value")
plt.grid(True, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 30-Day Plot
plt.figure(figsize=(12, 6))
plt.plot(dates_30, forecast_30, linestyle='-', color='darkorange')
plt.title(f"30-Day AQI Forecast (Starting {dates_30[0].strftime('%Y-%m-%d')})")
plt.xlabel("Date")
plt.ylabel("Predicted AQI Value")
plt.grid(True, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 365-Day Plot (Annual Trend)
plt.figure(figsize=(14, 7))
plt.plot(dates_365, forecast_365, linestyle='-', color='green')
plt.title(f"365-Day (Annual) AQI Forecast Trend")
plt.xlabel("Date")
plt.ylabel("Predicted AQI Value")
plt.grid(True, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------
# Save forecasts to CSV
# -------------------------
pd.DataFrame({
    'Date': dates_7,
    'Predicted_AQI': forecast_7
}).to_csv("forecast_7days.csv", index=False)

pd.DataFrame({
    'Date': dates_30,
    'Predicted_AQI': forecast_30
}).to_csv("forecast_30days.csv", index=False)

pd.DataFrame({
    'Date': dates_365,
    'Predicted_AQI': forecast_365
}).to_csv("forecast_365days.csv", index=False)

print("\nForecasts saved successfully:")
print(" - forecast_7days.csv")
print(" - forecast_30days.csv")
print(" - forecast_365days.csv")