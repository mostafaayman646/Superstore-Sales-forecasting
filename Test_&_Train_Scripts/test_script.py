import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from prophet.serialize import model_to_json, model_from_json
# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# --- Example Usage: Predict for a Specific Date (User Request) ---
# -----------------------------------------------------------------

#STEP 7 — Load Model and Make Forecast (Example)
# ======================================
model_filename = 'Model/sales_model.json'

print(f"\nLoading model from {model_filename}...")
with open(model_filename, 'r') as fin:
    loaded_model = model_from_json(fin.read())
print("Model loaded successfully.")

print("\n" + "="*50)
print("--- Example Usage: Predict Sales for a Specific Date ---")
print("="*50)

try:
    # 1. Get date input from the user
    #Example input to use 2020-05-31
    user_date_str = input("Enter a future date to predict (e.g., YYYY-MM-DD): ")
    
    # 2. Create a pandas DataFrame in the format Prophet requires
    # It MUST have a column named 'ds'
    future_date_df = pd.DataFrame({'ds': [user_date_str]})
    
    # 3. Ensure the 'ds' column is in datetime format
    # This will raise an error if the format is invalid, which our 'except' block will catch
    future_date_df['ds'] = pd.to_datetime(future_date_df['ds'])
    
    # 4. Use the trained model to make a prediction
    # model.predict() always takes a DataFrame
    forecast_for_date = loaded_model.predict(future_date_df)
    
    # 5. Display the prediction
    # The predicted value is in the 'yhat' column.
    predicted_sales = forecast_for_date['yhat'].iloc[0]
    
    print(f"\nPrediction for date: {user_date_str}")
    print(f"Predicted Sales ('yhat'): {predicted_sales:,.2f}")
    
    print("\n--- Prediction Details (with confidence interval) ---")
    print(forecast_for_date[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))
    
    print("\n*Important Note:")
    print("This model was trained on data aggregated by *month* (freq='M').")
    print("The prediction for a specific day represents the model's forecasted trend and")
    print("seasonality value for that point in time, based on the monthly patterns it learned.")


except Exception as e:
    print(f"\nAn error occurred. Please ensure your date is in a valid format (e.g., YYYY-MM-DD).")
    print(f"Error details: {e}")

print("\n" + "="*50)
print("--- End of Example ---")
print("="*50)