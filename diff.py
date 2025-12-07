import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

tf.get_logger().setLevel('ERROR')
sns.set_style("whitegrid")
DATA_DIR = "aqi_training_data" 
MODEL_PATH = "lstm_aqi_model.keras"
SCALER_PATH = "scaler.pkl"
LOOKBACK = 30          
EPOCHS = 60
BATCH_SIZE = 32
TEST_SPLIT_RATIO = 0.2
DATE_COL = "Date"
AQI_COL = "Overall AQI Value"

def load_and_preprocess_data(data_dir, date_col, aqi_col):
    """Loads, combines, cleans, and preprocesses all time series data."""
    print("⏳ Stage 1: Loading and Combining Data...")
    all_data = []
    
    CSV_FILES = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    
    for file_name in CSV_FILES:
        file_path = os.path.join(DATA_DIR, file_name) 
        print(file_path)
        
        try:
            df = pd.read_csv(file_path)
            if aqi_col in df.columns and date_col in df.columns:
                all_data.append(df[[date_col, aqi_col]])
            else:
                print(f"⚠️ Warning: Missing columns in {file_name}. Skipping.")
        except Exception as e:
            print(f"❌ Error loading {file_name}: {e}. Skipping.")

    if not all_data:
        raise FileNotFoundError("No valid AQI data files were loaded. Check paths and file contents.")

    df_combined = pd.concat(all_data, ignore_index=True)
    
    df_combined[date_col] = pd.to_datetime(df_combined[date_col], errors='coerce')
    df_combined = df_combined.set_index(date_col).sort_index()
    series = df_combined[aqi_col].apply(pd.to_numeric, errors='coerce')

    series = series.resample("D").mean()
    series = series.interpolate(limit_direction="both")
    series = series.dropna() 

    print(f"✅ Data Preparation Complete. Total continuous data points: {len(series)}")
    return series

def create_sequences(data, lookback):
    """Creates input (X) and output (y) sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, 0])
        y.append(data[i+lookback, 0])
    X = np.array(X)
    y = np.array(y)
    # Reshape X for LSTM input: (samples, timesteps, features)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def build_lstm_model(lookback):
    """Defines the LSTM model architecture."""
    model = Sequential([
        # LSTM layers are highly effective at learning temporal dependencies
        LSTM(64, return_sequences=True, input_shape=(lookback, 1), name='lstm_1'),
        Dropout(0.2, name='dropout_1'),
        LSTM(32, name='lstm_2'),
        Dropout(0.2, name='dropout_2'),
        Dense(1, name='output')
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def plot_training_history(history):
    """Plots the model's training and validation loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History (Loss vs. Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (Loss)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

def plot_aqi_distribution(series):
    """Plots the distribution of the overall AQI values."""
    plt.figure(figsize=(8, 6))
    sns.histplot(series.dropna(), kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Overall AQI Values (2020-2025)')
    plt.xlabel('Overall AQI Value')
    plt.ylabel('Frequency (Days)')
    plt.show()

def plot_test_vs_prediction(y_test_clean, y_pred_clean):
    """Plots the actual vs. predicted AQI values for the entire test set."""
    plt.figure(figsize=(14, 6))
    sns.lineplot(x=range(len(y_test_clean)), y=y_test_clean, label="Actual AQI", color='#1f77b4')
    sns.lineplot(x=range(len(y_pred_clean)), y=y_pred_clean, label="Predicted AQI", color='#ff7f0e', linestyle='--')
    plt.title('Actual vs Predicted AQI (Full Test Set)')
    plt.xlabel('Test Sample Index (Days)')
    plt.ylabel('AQI Value')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_year_vs_prediction(y_test_inv, y_pred_inv, series_index, year_to_plot=2025):
    """Plots actual vs. predicted for a specific year subset of the test data."""
    # Align the index of the test data back to the original series dates
    
    # 1. Get the original dates corresponding to the test set
    test_dates = series_index[-len(y_test_inv):] 
    
    # 2. Flatten and mask the data to keep only the clean, corresponding samples
    y_test_flat = y_test_inv.flatten()
    y_pred_flat = y_pred_inv.flatten()
    mask = np.isfinite(y_test_flat) & np.isfinite(y_pred_flat)
    
    y_test_clean = y_test_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    test_dates_clean = test_dates[mask]
    
    # 3. Filter for the target year (e.g., 2025)
    year_mask = (test_dates_clean.year == year_to_plot)
    y_test_year = y_test_clean[year_mask]
    y_pred_year = y_pred_clean[year_mask]
    dates_year = test_dates_clean[year_mask]

    if len(dates_year) == 0:
        print(f"⚠️ Warning: No test data available for the year {year_to_plot}.")
        return

    plt.figure(figsize=(14, 6))
    sns.lineplot(x=dates_year, y=y_test_year, label=f"Actual AQI ({year_to_plot})", color='green')
    sns.lineplot(x=dates_year, y=y_pred_year, label=f"Predicted AQI ({year_to_plot})", color='red', linestyle='--')
    plt.title(f'Actual vs Predicted AQI for the Year {year_to_plot}')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =================================================================
#             LSTM AQI FORECASTING MODEL PRESENTATION
# =================================================================
def run_presentation():
    """Main execution function structured for the presentation."""
    
    # --- Data Loading and Preprocessing ---
    try:
        series = load_and_preprocess_data(DATA_DIR, DATE_COL, AQI_COL)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        return
    
    print("\n--- Model Training Status Check ---")
    
    # --- Scaling ---
    values = series.values.reshape(-1, 1)
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        # --- Load Saved Model and Scaler ---
        print(f"✅ Found saved model and scaler. Loading for prediction...")
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # We still need the scaled data for the test/sequence creation
        scaled = scaler.transform(values)
        X, y = create_sequences(scaled, LOOKBACK)
        split_idx = int(len(X) * (1 - TEST_SPLIT_RATIO))
        X_test, y_test = X[split_idx:], y[split_idx:]
        
    else:
        # --- Train New Model ---
        print(f"⚠️ Saved model not found. Starting full training process...")
        
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
        
        # --- Sequence Creation ---
        X, y = create_sequences(scaled, LOOKBACK)
        print(f"Total sequences created: {len(X)}")
        
        # --- Train-Test Split ---
        split_idx = int(len(X) * (1 - TEST_SPLIT_RATIO))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f"Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")
        
        # --- Build & Train LSTM Model ---
        print("\n⏳ Stage 2: Building and Training LSTM Model...")
        model = build_lstm_model(LOOKBACK)
        
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=2, # Set to 2 for one line per epoch
            callbacks=[early_stop]
        )
        
        # --- Save Model and Scaler ---
        model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\n✅ Model and scaler saved successfully to {MODEL_PATH} and {SCALER_PATH}")
        
        # --- Plot Training History ---
        print("\n--- Training Visualization ---")
        plot_training_history(history)
        
    # --- Exploratory Data Analysis (EDA) ---
    print("\n--- Data Overview: AQI Value Distribution ---")
    plot_aqi_distribution(series)

    # =================================================================
    #             STAGE 3: Evaluation and Visualization
    # =================================================================
    
    print("\n⏳ Stage 3: Evaluating Performance on Test Data...")
    y_pred = model.predict(X_test, verbose=0)

    # --- Inverse Scaling and Cleaning ---
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    y_test_flat = y_test_inv.flatten()
    y_pred_flat = y_pred_inv.flatten()
    mask = np.isfinite(y_test_flat) & np.isfinite(y_pred_flat)
    y_test_clean = y_test_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    # --- Metrics Calculation ---
    if len(y_test_clean) > 0:
        mae = mean_absolute_error(y_test_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
        r2 = r2_score(y_test_clean, y_pred_clean)

        print("\n--- Evaluation Metrics (Test Set Performance) ---")
        print(f"Valid Test Samples = {len(y_test_clean)}")
        print(f"**MAE (Mean Absolute Error)** \t= **{mae:.2f}** (Avg. error in AQI points)")
        print(f"**RMSE (Root Mean Squared Error)** = **{rmse:.2f}** (Magnitude of error)")
        print(f"**R² (Coefficient of Determination)** = **{r2:.3f}** (Model fit quality)")

        # --- Test Set Plot ---
        print("\n--- Test Set: Actual vs. Predicted (Full Range) ---")
        plot_test_vs_prediction(y_test_clean, y_pred_clean)
        
        # --- Targeted 2025 Plot ---
        print("\n--- Key Subset: Actual vs. Predicted (Focus on 2025) ---")
        plot_year_vs_prediction(y_test_inv, y_pred_inv, series.index, year_to_plot=2025)
    else:
        print("\n--- Evaluation Metrics ---")
        print("WARNING: Test set is empty after cleaning NaN/Inf values. Cannot calculate metrics or plot.")
        
    # =================================================================
    #             STAGE 4: Future Forecasting
    # =================================================================
    
    def forecast_forward(model, last_sequence, steps):
        """Performs multi-step forecasting by iteratively predicting and updating the input sequence."""
        seq = last_sequence.copy()
        output = []
        # 
        for _ in range(steps):
            # Predict the next step using the current sequence
            pred = model.predict(seq.reshape(1, LOOKBACK, 1), verbose=0)[0][0]
            output.append(pred)
            # Update the sequence by dropping the oldest value and appending the new prediction
            seq = np.append(seq[1:], pred)
        return np.array(output)

    # Get the last 'LOOKBACK' scaled values from the *entire* dataset for the starting point
    last_seq = scaled[-LOOKBACK:, 0]

    print("\n⏳ Stage 4: Generating Future Forecasts (7, 30, and 365 days)...")
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
    
    # --- Plot Forecasts ---
    
    # 7-Day Plot (Short-term stability check)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=dates_7, y=forecast_7, marker='o', color='red')
    plt.title(f"7-Day AQI Forecast (Short-Term)")
    plt.xlabel("Date")
    plt.ylabel("Predicted AQI Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 30-Day Plot (Mid-term trend)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dates_30, y=forecast_30, color='darkorange')
    plt.title(f"30-Day AQI Forecast (Mid-Term)")
    plt.xlabel("Date")
    plt.ylabel("Predicted AQI Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 365-Day Plot (Annual Seasonality)
    plt.figure(figsize=(14, 7))
    sns.lineplot(x=dates_365, y=forecast_365, color='green')
    plt.title(f"365-Day (Annual) AQI Forecast Trend")
    plt.xlabel("Date")
    plt.ylabel("Predicted AQI Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Save forecasts to CSV ---
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

    print("\n✅ Final Output: Forecasts saved successfully to CSV files.")


if __name__ == "__main__":
    run_presentation()