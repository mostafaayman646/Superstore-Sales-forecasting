/* Add to existing App.css */

.description {
  color: #666;
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.confidence-interval {
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}

.interval-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.interval-label {
  font-weight: 600;
}

.interval-value {
  font-family: monospace;
}

/* Forecast Chart Styles */
.forecast-chart {
  background: white;
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.chart-controls {
  display: flex;
  gap: 1rem;
  align-items: end;
  margin-bottom: 2rem;
}

.chart-controls .form-group {
  margin-bottom: 0;
}

.chart-container {
  margin: 2rem 0;
}

.loading {
  text-align: center;
  padding: 2rem;
  color: #666;
}

.forecast-details {
  margin-top: 3rem;
}

.forecast-table {
  overflow-x: auto;
  margin-top: 1rem;
}

.forecast-table table {
  width: 100%;
  border-collapse: collapse;
  background: #f8f9fa;
}

.forecast-table th,
.forecast-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid #dee2e6;
}

.forecast-table th {
  background: #e9ecef;
  font-weight: 600;
}

.forecast-table tr:hover {
  background: #e9ecef;
}