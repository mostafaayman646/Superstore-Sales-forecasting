import warnings
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from itertools import product
import mlflow
import mlflow.sklearn
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import r2_score,root_mean_squared_error

warnings.filterwarnings("ignore")


class TimeSeriesAutoML:
    """
    Simple AutoML pipeline for time series forecasting.
    Trains HoltWinters, SARIMA, and Prophet models, tunes them, and saves the best one.
    """
    
    def __init__(self, experiment_name: str = "TimeSeries_AutoML"):
        self.experiment_name = experiment_name
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.train = None
        self.test = None
        
        # MLflow setup
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)
    
    # ============== Metrics ==============
    def _rmse(self, actual: pd.Series, predicted: pd.Series) -> float:
        
        return root_mean_squared_error(actual, predicted)
    
    def _r2(self, actual: pd.Series, predicted: pd.Series) -> float:
        
        return r2_score(actual, predicted)
    
    # ============== Data Prep ==============
    def _prepare_data(self, df: pd.DataFrame, train_size: float = 0.8):
        data = df[['Order Date', 'Sales']].copy()
        data['Order Date'] = pd.to_datetime(data['Order Date'])
        data.set_index('Order Date', inplace=True)
        monthly = data['Sales'].resample('M').sum()
        
        split = int(len(monthly) * train_size)
        self.train, self.test = monthly[:split], monthly[split:]
        return self.train, self.test
    
    # ============== Model Training ==============
    def _train_holtwinters(self) -> Dict[str, Any]:
        """Train HoltWinters with grid search tuning."""
        print("\n>>> Training HoltWinters (with tuning)...")
        
        param_grid = {
            'seasonal': ['add', 'mul'],
            'trend': ['add', 'mul'],
            'smoothing_level': [0.1, 0.2, 0.3, 0.5],
            'smoothing_trend': [0.1, 0.2, 0.3],
            'smoothing_seasonal': [0.1, 0.2, 0.3],
        }
        
        best_rmse, best_model, best_params = float('inf'), None, {}
        
        for seasonal, trend in product(param_grid['seasonal'], param_grid['trend']):
            for alpha in param_grid['smoothing_level']:
                for beta in param_grid['smoothing_trend']:
                    for gamma in param_grid['smoothing_seasonal']:
                        try:
                            model = ExponentialSmoothing(
                                self.train, seasonal=seasonal, 
                                seasonal_periods=12, trend=trend
                            )
                            fit = model.fit(
                                smoothing_level=alpha,
                                smoothing_trend=beta,
                                smoothing_seasonal=gamma
                            )
                            fc = fit.forecast(len(self.test))
                            rmse = self._rmse(self.test, fc)
                            
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_model = fit
                                best_params = {
                                    'seasonal': seasonal, 'trend': trend,
                                    'alpha': alpha, 'beta': beta, 'gamma': gamma
                                }
                        except:
                            continue
        
        fc = best_model.forecast(len(self.test))
        r2 = self._r2(self.test, fc)
        
        print(f"    Best params: {best_params}")
        print(f"    RMSE: {best_rmse:.4f} | R2: {r2:.4f}")
        
        return {
            'name': 'HoltWinters', 'model': best_model, 'params': best_params,
            'rmse': best_rmse, 'r2': r2, 'forecast': fc
        }
    
    def _train_sarima(self) -> Dict[str, Any]:
        """Train SARIMA with auto_arima tuning."""
        print("\n>>> Training SARIMA (with auto_arima tuning)...")
        
        monthly = pd.concat([self.train, self.test])
        
        # Auto-tune parameters
        auto_model = auto_arima(
            monthly, seasonal=True, m=12, 
            suppress_warnings=True, stepwise=True
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        
        print(f"    Auto-selected: order={order}, seasonal={seasonal_order}")
        
        # Fit final model
        model = sm.tsa.SARIMAX(monthly, order=order, seasonal_order=seasonal_order)
        fit = model.fit(disp=False)
        
        fitted = fit.fittedvalues
        rmse = self._rmse(monthly, fitted)
        r2 = self._r2(monthly, fitted)
        
        # Future forecast
        fc = fit.get_forecast(steps=12).predicted_mean
        
        print(f"    RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
        return {
            'name': 'SARIMA', 'model': fit, 
            'params': {'order': order, 'seasonal_order': seasonal_order},
            'rmse': rmse, 'r2': r2, 'forecast': fc
        }
    
    def _train_prophet(self) -> Dict[str, Any]:
        """Train Prophet with basic tuning."""
        print("\n>>> Training Prophet (with tuning)...")
        
        monthly = pd.concat([self.train, self.test])
        df_prophet = monthly.reset_index()
        df_prophet.columns = ['ds', 'y']
        
        # Try different seasonality modes
        best_rmse, best_model, best_params = float('inf'), None, {}
        
        for mode in ['additive', 'multiplicative']:
            try:
                model = Prophet(seasonality_mode=mode, yearly_seasonality=True)
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=0, freq='M')
                pred = model.predict(future)
                
                rmse = self._rmse(df_prophet['y'], pred['yhat'])
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = {'seasonality_mode': mode}
            except:
                continue
        
        fitted = best_model.predict(best_model.make_future_dataframe(periods=0, freq='M'))
        r2 = self._r2(df_prophet['y'], fitted['yhat'])
        
        # Future forecast
        future = best_model.make_future_dataframe(periods=12, freq='M')
        fc = best_model.predict(future)['yhat'].tail(12)
        
        print(f"    Best params: {best_params}")
        print(f"    RMSE: {best_rmse:.4f} | R2: {r2:.4f}")
        
        return {
            'name': 'Prophet', 'model': best_model, 'params': best_params,
            'rmse': best_rmse, 'r2': r2, 'forecast': fc
        }
    
    # ============== Main Pipeline ==============
    def run(self, df: pd.DataFrame, train_size: float = 0.8) -> Dict[str, Any]:
        """Run the full pipeline: prepare data, train all models, select best."""
        
        print(f"\n{'#'*60}")
        print(f"# TIMESERIES AUTOML PIPELINE")
        print(f"{'#'*60}")
        
        # Prepare data
        self._prepare_data(df, train_size)
        print(f"Data: {len(self.train)} train, {len(self.test)} test samples")
        
        # Train all models with MLflow logging
        models = [
            ('HoltWinters', self._train_holtwinters),
            ('SARIMA', self._train_sarima),
            ('Prophet', self._train_prophet)
        ]
        
        for name, train_fn in models:
            with mlflow.start_run(run_name=name):
                result = train_fn()
                self.results[name] = result
                
                # Log to MLflow
                for k, v in result['params'].items():
                    mlflow.log_param(k, str(v))
                mlflow.log_metric("rmse", result['rmse'])
                mlflow.log_metric("r2", result['r2'])
        
        # Select best model (lowest RMSE)
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['r2'])
        self.best_model = self.results[self.best_model_name]
        
        # Display results
        self._display_results()
        
        # Save best model
        self._save_best_model()
        
        return self.best_model
    
    def _display_results(self):
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'RMSE':<15} {'R2':<10}")
        print("-" * 45)
        
        for name, r in sorted(self.results.items(), key=lambda x: x[1]['rmse']):
            marker = " *** BEST" if name == self.best_model_name else ""
            print(f"{name:<20} {r['rmse']:<15.4f} {r['r2']:<10.4f}{marker}")
        
        print(f"\n>>> Best Model: {self.best_model_name}")
        print(f"    Params: {self.best_model['params']}")
    
    def _save_best_model(self, path: str = "Models/best_model"):
        """Save best model info as JSON and log to MLflow."""
        
        # Prepare JSON-serializable data
        model_info = {
            'name': self.best_model['name'],
            'params': {k: str(v) for k, v in self.best_model['params'].items()},
            'rmse': float(self.best_model['rmse']),
            'r2': float(self.best_model['r2']),
            'forecast': self.best_model['forecast'].tolist() if hasattr(self.best_model['forecast'], 'tolist') else list(self.best_model['forecast'])
        }
        
        # Save JSON
        with open(f"{path}.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Register in MLflow
        with mlflow.start_run(run_name="Best_Model"):
            mlflow.log_param("best_model", self.best_model_name)
            mlflow.log_metric("best_rmse", self.best_model['rmse'])
            mlflow.log_metric("best_r2", self.best_model['r2'])
            mlflow.log_artifact(f"{path}.json")
            
            # Log actual model object
            if self.best_model_name == "Prophet":
                mlflow.prophet.log_model(self.best_model['model'], "model")
            else:
                mlflow.statsmodels.log_model(self.best_model['model'], "model")
        
        print(f"\n>>> Model saved to: {path}.json")
        print(f">>> MLflow URI: {mlflow.get_tracking_uri()}")

# ============== Usage ==============
if __name__ == "__main__":
    from Data_helper import preprocessing
    
    df = preprocessing()
    
    pipeline = TimeSeriesAutoML(experiment_name="Sales_Forecasting")
    best = pipeline.run(df, train_size=0.8)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)