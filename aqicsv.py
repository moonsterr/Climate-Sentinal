import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("aqidaily2025.csv")

# Convert Date column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Columns to convert to numeric
numeric_cols = ['Overall AQI Value', 'CO', 'Ozone', 'PM10', 'PM25', 'NO2']

# Convert numeric columns, replace non-numeric values with NaN
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Optionally, fill missing values with column mean (or drop rows)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print(df.info())

# -----------------------
# Trend of AQI over time
# -----------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Overall AQI Value'], label='AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('Daily AQI Trend')
plt.legend()
plt.show()

# -----------------------
# Correlation heatmap
# -----------------------
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between pollutants and AQI')
plt.show()

# -----------------------
# Average AQI by site
# -----------------------
site_avg = df.groupby('Site Name (of Overall AQI)')['Overall AQI Value'].mean().sort_values(ascending=False)
print(site_avg)

# -----------------------
# AQI trends per site
# -----------------------
plt.figure(figsize=(12,6))
for site in df['Site Name (of Overall AQI)'].unique():
    df[df['Site Name (of Overall AQI)']==site]['Overall AQI Value'].plot(label=site)
plt.legend()
plt.title("AQI Trend by Site")
plt.show()

# -----------------------
# Top 10 AQI values
# -----------------------
top_aqi = df.sort_values('Overall AQI Value', ascending=False).head(10)
print(top_aqi[['Overall AQI Value','Main Pollutant']])

# -----------------------
# Monthly averages
# -----------------------
monthly_avg = df.resample('M').mean()
monthly_avg[['Overall AQI Value','PM25','PM10','Ozone']].plot(kind='bar', figsize=(12,6))
plt.title('Monthly Average AQI & Pollutants')
plt.show()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("aqidaily2025.csv")

# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date')
features = ['CO','Ozone','PM10','PM25','NO2']
# target = 'Overall AQI Value'

# df[features + [target]] = df[features + [target]].apply(pd.to_numeric, errors='coerce')

# print(df)
# df[features + [target]] = df[features + [target]].fillna(df[features + [target]].mean())
# print(df)

# X = df[features]
# y = df[target]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_test, X_train, y_train, y_test)

# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R^2 Score:", r2)

# importance = pd.Series(model.feature_importances_, index=features)
# importance.sort_values().plot(kind='barh', figsize=(8,5), color='green')
# plt.title("Feature Importance for AQI Prediction")
# plt.show()

# plt.figure(figsize=(12,6))
# plt.plot(y_test.values, label='Actual', marker='o')
# plt.plot(y_pred, label='Predicted', marker='x')
# plt.title('AQI: Actual vs Predicted')
# plt.legend()
# plt.show()
