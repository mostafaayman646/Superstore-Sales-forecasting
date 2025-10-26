import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from prophet import Prophet
from Data_helper import *
from prophet.serialize import model_to_json, model_from_json
# Suppress warnings
warnings.filterwarnings("ignore")

df = preprocessing()

df_copy = df[['Order Date', 'Sales']].copy()

df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date'])

monthly_data = df_copy.resample('M', on='Order Date').sum()

# Rename columns as 'ds' and 'y' as required by Prophet
monthly_data.reset_index(inplace=True)
monthly_data = monthly_data.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(monthly_data)

# Create a future dataframe for the next 12 months
future = model.make_future_dataframe(periods=12, freq='M')

# Make forecasts for the future
forecast = model.predict(future)

# Extract the actual data and forecast for the next 12 months
actual_data = monthly_data['y']
fitted_data = forecast['yhat'][:len(monthly_data)]  # Fitted values for the historical data
forecast_data = forecast['yhat'][len(monthly_data):]

# Calculate R-squared score for the fitted data
r2_fitted = 1 - np.sum((actual_data - fitted_data) ** 2) / np.sum((actual_data - actual_data.mean()) ** 2)
# Display R-squared score for the fitted data
print(f"R-squared score for fitted data: {r2_fitted:.2f}")

# --- ADDED: Save the model ---
model_filename = 'Model/sales_model.json'
print(f"Saving final model to {model_filename}...")
with open(model_filename, 'w') as fout:
    fout.write(model_to_json(model))
print("Model saved successfully.")