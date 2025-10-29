from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import numpy as np
from datetime import datetime
import os
from prophet.serialize import model_from_json

app = Flask(__name__)
CORS(app)

# Load model and metadata
try:
    model_filename = 'Model/sales_model.json'
    with open(model_filename, 'r') as fin:
        model = model_from_json(fin.read())
    
    with open('Model/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Load training data for reference
    training_data = pd.read_csv('Model/monthly_training_data.csv')
    training_data['ds'] = pd.to_datetime(training_data['ds'])
    
    print("Prophet model loaded successfully!")
    print(f"Model trained on data from {model_metadata['training_data_range']['start']} to {model_metadata['training_data_range']['end']}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_metadata = {}
    training_data = None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'Prophet',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict_sales():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'date' not in data:
            return jsonify({'error': 'Date is required'}), 400
        
        # Parse the input date
        user_date = pd.to_datetime(data['date'])
        
        # Create future dataframe for prediction
        future_date_df = pd.DataFrame({'ds': [user_date]})
        
        # Make prediction
        forecast = model.predict(future_date_df)
        
        prediction_result = {
            'date': user_date.strftime('%Y-%m-%d'),
            'predicted_sales': float(forecast['yhat'].iloc[0]),
            'prediction_lower': float(forecast['yhat_lower'].iloc[0]),
            'prediction_upper': float(forecast['yhat_upper'].iloc[0]),
            'prediction_date': datetime.now().isoformat()
        }
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/forecast', methods=['POST'])
def forecast_range():
    """Generate forecast for a range of dates"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        periods = data.get('periods', 12)  # Default to 12 months
        freq = data.get('freq', 'M')  # Default to monthly frequency
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Get only future predictions (beyond training data)
        future_forecast = forecast[forecast['ds'] > training_data['ds'].max()].tail(periods)
        
        # Convert to list of dictionaries for JSON response
        forecast_data = []
        for _, row in future_forecast.iterrows():
            forecast_data.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'predicted_sales': float(row['yhat']),
                'prediction_lower': float(row['yhat_lower']),
                'prediction_upper': float(row['yhat_upper'])
            })
        
        return jsonify({
            'forecast': forecast_data,
            'periods': periods,
            'frequency': freq
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'metadata': model_metadata,
        'model_type': 'Prophet Time Series',
        'training_data_points': len(training_data) if training_data is not None else 0
    })

@app.route('/api/historical-data', methods=['GET'])
def get_historical_data():
    """Get historical training data"""
    if training_data is None:
        return jsonify({'error': 'Training data not available'}), 500
    
    historical_data = []
    for _, row in training_data.iterrows():
        historical_data.append({
            'date': row['ds'].strftime('%Y-%m-%d'),
            'sales': float(row['y'])
        })
    
    return jsonify({
        'historical_data': historical_data,
        'data_range': model_metadata.get('training_data_range', {})
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)