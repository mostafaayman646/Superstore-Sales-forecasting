from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import numpy as np
from datetime import datetime
import os
import mlflow
import mlflow.prophet
from prophet import Prophet

app = Flask(__name__)
CORS(app)

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "Sales_Forecasting"

# Global variables
model = None
model_metadata = {}
training_data = None

def load_best_model_from_mlflow():
    """Load the best model and metadata from MLflow"""
    global model, model_metadata, training_data
    
    try:
        # Get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return False
        
        # Search for the best run (highest R2 score)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'Best_Model'",
            order_by=["metrics.best_r2 DESC"]
        )
        
        if runs.empty:
            print("No 'Best_Model' run found in MLflow")
            # Try to find any Prophet run with good R2 score
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runName = 'Prophet'",
                order_by=["metrics.r2 DESC"]
            )
        
        if runs.empty:
            print("No suitable model runs found in MLflow")
            return False
        
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        
        print(f"Loading model from run: {run_id}")
        print(f"Model performance - R2: {best_run.get('metrics.r2', best_run.get('metrics.best_r2', 'N/A'))}")
        
        # Load the model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.prophet.load_model(model_uri)
        
        # Create metadata from MLflow run data
        model_metadata = {
            "model_type": "Prophet",
            "training_data_range": {
                "start": "2015-01-31",  # You might want to extract this from run params
                "end": "2018-12-31"     # or from the actual training data
            },
            "performance": {
                "r2_score": float(best_run.get('metrics.r2', best_run.get('metrics.best_r2', 0.0)))
            },
            "training_date": best_run.start_time.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(best_run.start_time) else "Unknown",
            "data_frequency": "M",
            "target_variable": "Sales",
            "mlflow_run_id": run_id,
            "model_params": {
                k: v for k, v in best_run.to_dict().items() 
                if k.startswith('params.') and pd.notna(v)
            }
        }
        
        # Load or create training data reference
        # Since MLflow might not store the original training data, we'll use the model's history
        if model.history is not None:
            training_data = model.history[['ds', 'y']].copy()
        else:
            # Fallback: try to load from the original file
            try:
                training_data = pd.read_csv('Model/monthly_training_data.csv')
                training_data['ds'] = pd.to_datetime(training_data['ds'])
            except:
                print("Warning: Could not load training data")
                training_data = None
        
        print("Prophet model loaded successfully from MLflow!")
        print(f"Model R2 score: {model_metadata['performance']['r2_score']}")
        print(f"MLflow Run ID: {run_id}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return False

# Load model on startup
try:
    if not load_best_model_from_mlflow():
        print("Falling back to file-based model loading...")
        # Fallback to original file-based loading
        from prophet.serialize import model_from_json
        
        model_filename = 'Model/sales_model.json'
        with open(model_filename, 'r') as fin:
            model = model_from_json(fin.read())
        
        with open('Model/model_metadata.json', 'r') as f:
            model_metadata = json.load(f)
        
        training_data = pd.read_csv('Model/monthly_training_data.csv')
        training_data['ds'] = pd.to_datetime(training_data['ds'])
        
        print("Prophet model loaded from files successfully!")
        
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
        'mlflow_loaded': 'mlflow_run_id' in model_metadata if model_metadata else False,
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
            'prediction_date': datetime.now().isoformat(),
            'mlflow_run_id': model_metadata.get('mlflow_run_id', 'file_based')
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
        if training_data is not None:
            future_forecast = forecast[forecast['ds'] > training_data['ds'].max()].tail(periods)
        else:
            # If we don't have training data, just take the last 'periods' forecasts
            future_forecast = forecast.tail(periods)
        
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
            'frequency': freq,
            'mlflow_run_id': model_metadata.get('mlflow_run_id', 'file_based')
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
        'training_data_points': len(training_data) if training_data is not None else 0,
        'loaded_from_mlflow': 'mlflow_run_id' in model_metadata
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
        'data_range': model_metadata.get('training_data_range', {}),
        'mlflow_run_id': model_metadata.get('mlflow_run_id', 'file_based')
    })

@app.route('/api/mlflow-runs', methods=['GET'])
def get_mlflow_runs():
    """Get information about available MLflow runs"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return jsonify({'error': f'Experiment {experiment_name} not found'}), 404
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2 DESC"]
        )
        
        run_info = []
        for _, run in runs.iterrows():
            run_info.append({
                'run_id': run.run_id,
                'run_name': run.get('tags.mlflow.runName', 'Unknown'),
                'r2_score': run.get('metrics.r2', run.get('metrics.best_r2', None)),
                'rmse': run.get('metrics.rmse', run.get('metrics.best_rmse', None)),
                'start_time': run.start_time.isoformat() if pd.notna(run.start_time) else None,
                'status': run.status
            })
        
        return jsonify({
            'experiment_name': experiment_name,
            'runs': run_info,
            'current_run_id': model_metadata.get('mlflow_run_id', None)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)