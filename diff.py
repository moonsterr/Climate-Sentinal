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

# --- GLOBAL CONFIGURATION ---
GLOBAL_DATA_DIR = "aqi_training_data"  # Data for Global Model Training
CITY_DATA_FILE = "beir_data.csv"       # New City Data for Forecasting (Example)
GLOBAL_MODEL_PATH = "lstm_global_aqi_model.keras"
GLOBAL_SCALER_PATH = "global_scaler.pkl"
LOOKBACK = 30
EPOCHS = 60
BATCH_SIZE = 32
TEST_SPLIT_RATIO = 0.2
DATE_COL = "Date"
AQI_COL = "Overall AQI Value"
CITY_NAME = "Beirut" # Name of the city for the forecast

# --- PLOTTING FUNCTIONS (Kept as is) ---
# plot_training_history, plot_aqi_distribution, plot_test_vs_prediction, plot_year_vs_prediction
# ... (These functions are omitted here for brevity, assume they are included)

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
    plt.title('Distribution of Overall AQI Values (Global Training Data)')
    plt.xlabel('Overall AQI Value')
    plt.ylabel('Frequency (Days)')
    plt.show()

def plot_city_forecast(dates, forecast, city_name, period):
    """Plots the city-specific future forecast."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dates, y=forecast, color='purple', linewidth=2)
    plt.title(f'{city_name} - {period} AQI Forecast Trend')
    plt.xlabel("Date")
    plt.ylabel("Predicted AQI Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- UTILITY FUNCTIONS (Modified for Reusability) ---

def load_and_preprocess_data(file_source, date_col, aqi_col, is_global_dir=True):
    """Loads, combines (if dir), cleans, and preprocesses time series data."""
    print(f"⏳ Loading data from: {file_source}...")
    all_data = []

    if is_global_dir:
        # Load multiple files from a directory for global training
        CSV_FILES = [f for f in os.listdir(file_source) if f.endswith(".csv")]
        for file_name in CSV_FILES:
            file_path = os.path.join(file_source, file_name)
            try:
                df = pd.read_csv(file_path)
                if aqi_col in df.columns and date_col in df.columns:
                    all_data.append(df[[date_col, aqi_col]])
            except Exception as e:
                print(f"❌ Error loading {file_name}: {e}. Skipping.")
    else:
        # Load a single file for city-specific forecasting
        try:
            df = pd.read_csv(file_source)
            if aqi_col in df.columns and date_col in df.columns:
                all_data.append(df[[date_col, aqi_col]])
            else:
                 print(f"⚠️ Warning: Missing columns in {file_source}. Cannot proceed.")
        except Exception as e:
            raise FileNotFoundError(f"❌ Error loading city data {file_source}: {e}")

    if not all_data:
        raise FileNotFoundError("No valid AQI data files were loaded. Check paths and file contents.")

    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined[date_col] = pd.to_datetime(df_combined[date_col], errors='coerce')
    df_combined = df_combined.set_index(date_col).sort_index()

    # Apply to_numeric *before* resample/interpolate
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
    # This architecture remains the same for the global model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1), name='lstm_1'),
        Dropout(0.2, name='dropout_1'),
        LSTM(32, name='lstm_2'),
        Dropout(0.2, name='dropout_2'),
        Dense(1, name='output')
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

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

# =================================================================
#               GLOBAL TRAINING MODE
# =================================================================

def train_global_model():
    """Trains and saves the global AQI forecasting model and scaler."""
    print("=================================================================")
    print("        STAGE 1: GLOBAL MODEL TRAINING AND SAVING")
    print("=================================================================")

    # --- Data Loading and Preprocessing ---
    try:
        series = load_and_preprocess_data(GLOBAL_DATA_DIR, DATE_COL, AQI_COL, is_global_dir=True)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        return

    # --- Data Overview ---
    print("\n--- Data Overview: AQI Value Distribution (Global) ---")
    plot_aqi_distribution(series)

    # --- Scaling ---
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values) # FIT on GLOBAL data

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
        verbose=2,
        callbacks=[early_stop]
    )

    # --- Save Model and Scaler ---
    model.save(GLOBAL_MODEL_PATH)
    with open(GLOBAL_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n✅ Global model and scaler saved successfully to {GLOBAL_MODEL_PATH} and {GLOBAL_SCALER_PATH}")
    print("\n--- Training Visualization ---")
    plot_training_history(history)

    # --- Evaluation (Optional but good practice) ---
    # The evaluation logic from the original script can be placed here to check global model performance.


# =================================================================
#               CITY-SPECIFIC FORECASTING MODE
# =================================================================

def forecast_city_data(city_data_file, city_name):
    """Loads a trained global model/scaler and forecasts AQI for a specific city."""
    print("\n=================================================================")
    print(f"        STAGE 2: CITY-SPECIFIC FORECASTING FOR {city_name.upper()}")
    print("=================================================================")

    # --- Load Global Model and Scaler ---
    if not os.path.exists(GLOBAL_MODEL_PATH) or not os.path.exists(GLOBAL_SCALER_PATH):
        print(f"❌ Error: Global model or scaler not found. Run 'train_global_model()' first.")
        return

    model = load_model(GLOBAL_MODEL_PATH)
    with open(GLOBAL_SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Loaded Global Model and Scaler for {city_name} forecasting.")

    # --- Load and Preprocess City Data ---
    try:
        city_series = load_and_preprocess_data(city_data_file, DATE_COL, AQI_COL, is_global_dir=False)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        return

    # --- Scale City Data (Using Global Scaler) ---
    city_values = city_series.values.reshape(-1, 1)
    # The crucial step: use the loaded global scaler's TRANSFORM method
    scaled_city_data = scaler.transform(city_values)
    print("✅ City data scaled using the Global Model's scaler.")

    # --- Prepare Last Sequence for Forecast ---
    if len(scaled_city_data) < LOOKBACK:
        print(f"❌ Error: City data must have at least {LOOKBACK} data points for forecasting.")
        return

    # Get the last 'LOOKBACK' scaled values from the city's historical data
    last_city_seq = scaled_city_data[-LOOKBACK:, 0]
    
    # --- Generate Forecasts ---
    print("\n⏳ Generating Future Forecasts (7, 30, and 365 days)...")
    
    # 1. 7-Day Forecast
    forecast_7 = forecast_forward(model, last_city_seq, 7)
    forecast_7 = scaler.inverse_transform(forecast_7.reshape(-1, 1)).flatten()
    
    # 2. 30-Day Forecast
    # Note: We must re-calculate the last_city_seq before each long-term forecast 
    # if we want the intermediate forecasts (7-day, 30-day) to be generated 
    # from the exact same starting point.
    forecast_30 = forecast_forward(model, last_city_seq, 30) 
    forecast_30 = scaler.inverse_transform(forecast_30.reshape(-1, 1)).flatten()
    
    # 3. 365-Day Forecast
    forecast_365 = forecast_forward(model, last_city_seq, 365)
    forecast_365 = scaler.inverse_transform(forecast_365.reshape(-1, 1)).flatten()
    
    # --- Date and Output Preparation ---
    last_date = city_series.index[-1]
    
    dates_7 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    dates_30 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    dates_365 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)
    
    # --- Plot Forecasts ---
    plot_city_forecast(dates_7, forecast_7, city_name, "7-Day")
    plot_city_forecast(dates_30, forecast_30, city_name, "30-Day")
    plot_city_forecast(dates_365, forecast_365, city_name, "365-Day (Annual)")

    # --- Save forecasts to CSV ---
    pd.DataFrame({
        'Date': dates_7,
        'Predicted_AQI': forecast_7
    }).to_csv(f"{city_name.lower()}_forecast_7days.csv", index=False)

    pd.DataFrame({
        'Date': dates_30,
        'Predicted_AQI': forecast_30
    }).to_csv(f"{city_name.lower()}_forecast_30days.csv", index=False)

    pd.DataFrame({
        'Date': dates_365,
        'Predicted_AQI': forecast_365
    }).to_csv(f"{city_name.lower()}_forecast_365days.csv", index=False)

    print(f"\n✅ Final Output: Forecasts for {city_name} saved successfully to CSV files.")


if __name__ == "__main__":
    # 1. Train the Global Model (Only run once or when new data is added)
    train_global_model()
    
    # 2. Use the Global Model for City-Specific Forecasting
    forecast_city_data(CITY_DATA_FILE, CITY_NAME)