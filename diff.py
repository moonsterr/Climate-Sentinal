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

# --- CONFIGURATION (Now Class-level or passed in) ---
GLOBAL_DATA_DIR = "aqi_training_data"
CITY_DATA_FILE = "beir_data.csv"
GLOBAL_MODEL_PATH = "lstm_global_aqi_model.keras"
GLOBAL_SCALER_PATH = "global_scaler.pkl"
LOOKBACK = 30
EPOCHS = 60
BATCH_SIZE = 32
TEST_SPLIT_RATIO = 0.2
DATE_COL = "Date"
AQI_COL = "Overall AQI Value"
CITY_NAME = "Beirut" 
# =================================================================
#                   AQI FORECASTER CLASS
# =================================================================
class AqiForecaster:
    """
    Manages the training of a global LSTM model and city-specific AQI forecasting.
    """
    def __init__(self, lookback, epochs, batch_size, date_col, aqi_col):
        self.LOOKBACK = lookback
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.DATE_COL = date_col
        self.AQI_COL = aqi_col
        self.model = None
        self.scaler = None

    # --- Utility Methods ---

    def _load_and_preprocess_data(self, file_source, is_global_dir=True):
        """Loads, cleans, resamples, and interpolates time series data."""
        print(f"⏳ Loading data from: {file_source}...")
        all_data = []

        if is_global_dir:
            CSV_FILES = [f for f in os.listdir(file_source) if f.endswith(".csv")]
            file_list = [os.path.join(file_source, f) for f in CSV_FILES]
        else:
            file_list = [file_source]

        for file_path in file_list:
            try:
                df = pd.read_csv(file_path)
                if self.AQI_COL in df.columns and self.DATE_COL in df.columns:
                    all_data.append(df[[self.DATE_COL, self.AQI_COL]])
            except Exception as e:
                print(f"❌ Error loading {os.path.basename(file_path)}: {e}. Skipping.")

        if not all_data:
            raise FileNotFoundError("No valid AQI data files were loaded. Check paths and file contents.")

        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined[self.DATE_COL] = pd.to_datetime(df_combined[self.DATE_COL], errors='coerce')
        df_combined = df_combined.set_index(self.DATE_COL).sort_index()

        series = df_combined[self.AQI_COL].apply(pd.to_numeric, errors='coerce')
        series = series.resample("D").mean()
        series = series.interpolate(limit_direction="both")
        series = series.dropna()

        print(f"✅ Data Preparation Complete. Total continuous data points: {len(series)}")
        return series

    def _create_sequences(self, data):
        """Creates input (X) and output (y) sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.LOOKBACK):
            # X is the LOOKBACK sequence
            X.append(data[i:i + self.LOOKBACK, 0])
            # y is the value immediately following the sequence
            y.append(data[i + self.LOOKBACK, 0])
        X = np.array(X)
        y = np.array(y)
        # Reshape X for LSTM input: (samples, timesteps, features)
        return X.reshape((X.shape[0], X.shape[1], 1)), y

    def _build_model(self):
        """Defines the LSTM model architecture."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.LOOKBACK, 1), name='lstm_1'),
            Dropout(0.2, name='dropout_1'),
            LSTM(32, name='lstm_2'),
            Dropout(0.2, name='dropout_2'),
            Dense(1, name='output')
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def _forecast_forward(self, last_sequence, steps):
        """Performs multi-step forecasting by iteratively predicting and updating the input sequence."""
        # This function implements the recursive multi-step forecasting strategy.
        
        seq = last_sequence.copy()
        output = []
        for _ in range(steps):
            # Predict the next step using the current sequence
            pred = self.model.predict(seq.reshape(1, self.LOOKBACK, 1), verbose=0)[0][0]
            output.append(pred)
            # Update the sequence by dropping the oldest value and appending the new prediction
            seq = np.append(seq[1:], pred)
        return np.array(output)

    # --- Core Methods: Global Training ---
    
    def train_global_model(self, data_dir, model_path, scaler_path):
        """Trains and saves the global AQI forecasting model and scaler."""
        print("=================================================================")
        print("        STAGE 1: GLOBAL MODEL TRAINING AND SAVING 🌍")
        print("=================================================================")

        try:
            series = self._load_and_preprocess_data(data_dir, is_global_dir=True)
        except FileNotFoundError as e:
            print(f"\nFATAL ERROR: {e}")
            return

        values = series.values.reshape(-1, 1)
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(values) # FIT on GLOBAL data
        
        # Save the fitted scaler immediately
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        X, y = self._create_sequences(scaled)
        split_idx = int(len(X) * (1 - TEST_SPLIT_RATIO))
        X_train, y_train = X[:split_idx], y[:split_idx]
        
        print("\n⏳ Building and Training LSTM Model...")
        self.model = self._build_model()
        
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_split=0.1,
            verbose=2,
            callbacks=[early_stop]
        )
        
        self.model.save(model_path)
        print(f"\n✅ Global model saved to {model_path}")
        print(f"✅ Global scaler saved to {scaler_path}")
        self._plot_training_history(history)
        
    def _plot_training_history(self, history):
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

    # --- Core Methods: City Forecasting ---

    def load_model_and_scaler(self, model_path, scaler_path):
        """Loads a pre-trained model and scaler."""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"❌ Error: Model or scaler not found at {model_path} or {scaler_path}.")
            return False

        self.model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        return True

    def forecast_city(self, city_data_file, city_name, forecast_steps=[7, 30, 365]):
        """Generates and saves forecasts for a specific city using the global model."""
        if self.model is None or self.scaler is None:
            print("❌ Error: Model and/or scaler not loaded. Cannot forecast.")
            return

        print("\n=================================================================")
        print(f"        STAGE 2: CITY-SPECIFIC FORECASTING FOR {city_name.upper()} 🏙️")
        print("=================================================================")

        try:
            city_series = self._load_and_preprocess_data(city_data_file, is_global_dir=False)
        except FileNotFoundError as e:
            print(f"\nFATAL ERROR: {e}")
            return

        # --- Scale City Data (Using Global Scaler) ---
        city_values = city_series.values.reshape(-1, 1)
        scaled_city_data = self.scaler.transform(city_values)
        print("✅ City data scaled using the Global Model's scaler.")

        if len(scaled_city_data) < self.LOOKBACK:
            print(f"❌ Error: City data must have at least {self.LOOKBACK} data points for forecasting.")
            return

        # Get the last 'LOOKBACK' scaled values from the city's historical data
        last_city_seq = scaled_city_data[-self.LOOKBACK:, 0]
        last_date = city_series.index[-1]
        
        print("\n⏳ Generating Future Forecasts...")

        for steps in forecast_steps:
            forecast_output = self._forecast_forward(last_city_seq, steps)
            
            # Inverse-transform to get the final AQI values
            forecast_aqi = self.scaler.inverse_transform(forecast_output.reshape(-1, 1)).flatten()
            
            # Create dates
            dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
            
            # Plot and Save
            self._plot_city_forecast(dates, forecast_aqi, city_name, f"{steps}-Day")
            self._save_forecast(dates, forecast_aqi, city_name, steps)

        print(f"\n✅ Final Output: All forecasts for {city_name} completed.")

    def _plot_city_forecast(self, dates, forecast, city_name, period):
        """Plots the city-specific future forecast."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=dates, y=forecast, color='purple', linewidth=2)
        plt.title(f'{city_name} - {period} AQI Forecast Trend')
        plt.xlabel("Date")
        plt.ylabel("Predicted AQI Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _save_forecast(self, dates, forecast, city_name, steps):
        """Saves the forecast data to a CSV file."""
        df_forecast = pd.DataFrame({
            'Date': dates,
            'Predicted_AQI': forecast
        })
        filename = f"{city_name.lower().replace(' ', '_')}_forecast_{steps}days.csv"
        df_forecast.to_csv(filename, index=False)
        print(f"💾 Saved {steps}-day forecast to {filename}")

# =================================================================
#               MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    # 1. Initialize the Forecaster
    forecaster = AqiForecaster(
        lookback=LOOKBACK,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        date_col=DATE_COL,
        aqi_col=AQI_COL
    )

    # 2. Global Training Mode (Run this only if the model doesn't exist)
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print("Global model not found. Starting training...")
        forecaster.train_global_model(GLOBAL_DATA_DIR, GLOBAL_MODEL_PATH, GLOBAL_SCALER_PATH)
    else:
        print("Global model already exists. Skipping training.")

    # 3. City-Specific Forecasting Mode (Run this to generate predictions for a new city)
    if forecaster.load_model_and_scaler(GLOBAL_MODEL_PATH, GLOBAL_SCALER_PATH):
        forecaster.forecast_city(CITY_DATA_FILE, CITY_NAME, forecast_steps=[7, 30, 365])