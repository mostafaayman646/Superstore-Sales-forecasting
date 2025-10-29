import React, { useState } from 'react';
import { predictSales } from '../services/api';
import { format } from 'date-fns';

const SalesPredictor = () => {
  const [date, setDate] = useState(format(new Date(), 'yyyy-MM-dd'));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const result = await predictSales(date);
      setPrediction(result);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="sales-predictor">
      <h2>Sales Prediction</h2>
      <p className="description">
        Enter a future date to predict sales using the Prophet time series model.
        The model was trained on monthly SuperStore sales data.
      </p>
      
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-group">
          <label htmlFor="date-input">Select Date:</label>
          <input
            id="date-input"
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            required
            min={format(new Date(), 'yyyy-MM-dd')}
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Sales'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}

      {prediction && (
        <div className="prediction-result">
          <h3>Prediction Result</h3>
          <div className="prediction-card">
            <div className="prediction-value">
              ${prediction.predicted_sales.toFixed(2)}
            </div>
            <p>Predicted Sales for {prediction.date}</p>
            
            <div className="confidence-interval">
              <div className="interval-item">
                <span className="interval-label">Lower Bound:</span>
                <span className="interval-value">${prediction.prediction_lower.toFixed(2)}</span>
              </div>
              <div className="interval-item">
                <span className="interval-label">Upper Bound:</span>
                <span className="interval-value">${prediction.prediction_upper.toFixed(2)}</span>
              </div>
            </div>
            
            <small>Predicted on: {new Date(prediction.prediction_date).toLocaleString()}</small>
          </div>
        </div>
      )}
    </div>
  );
};

export default SalesPredictor;