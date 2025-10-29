import React from 'react';

const ModelInfo = ({ modelInfo }) => {
  if (!modelInfo) {
    return <div className="model-info">Loading model information...</div>;
  }

  const { metadata, model_type, training_data_points } = modelInfo;

  return (
    <div className="model-info">
      <h2>Model Information</h2>
      
      <div className="info-section">
        <h3>Model Details</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="label">Model Type:</span>
            <span className="value">{model_type}</span>
          </div>
          <div className="info-item">
            <span className="label">Training Data Points:</span>
            <span className="value">{training_data_points}</span>
          </div>
          <div className="info-item">
            <span className="label">Data Frequency:</span>
            <span className="value">{metadata.data_frequency}</span>
          </div>
          <div className="info-item">
            <span className="label">Target Variable:</span>
            <span className="value">{metadata.target_variable}</span>
          </div>
        </div>
      </div>

      <div className="info-section">
        <h3>Training Data Range</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="label">Start Date:</span>
            <span className="value">{metadata.training_data_range.start}</span>
          </div>
          <div className="info-item">
            <span className="label">End Date:</span>
            <span className="value">{metadata.training_data_range.end}</span>
          </div>
        </div>
      </div>

      <div className="info-section">
        <h3>Model Performance</h3>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{metadata.performance.r2_score.toFixed(3)}</div>
            <div className="metric-label">R² Score</div>
          </div>
        </div>
        <p className="performance-note">
          R² score indicates how well the model fits the training data. 
          Higher values (closer to 1.0) indicate better fit.
        </p>
      </div>

      <div className="info-section">
        <h3>Model Training</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="label">Training Date:</span>
            <span className="value">{metadata.training_date}</span>
          </div>
        </div>
      </div>

      <div className="info-section">
        <h3>About Prophet</h3>
        <p>
          Prophet is Facebook's open-source time series forecasting procedure that handles 
          daily observations with seasonal patterns. It's designed to automatically detect 
          changepoints and handle holidays and other special events.
        </p>
      </div>
    </div>
  );
};

export default ModelInfo;