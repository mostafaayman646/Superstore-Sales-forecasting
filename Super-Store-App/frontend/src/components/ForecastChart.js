import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getForecast, getHistoricalData } from '../services/api';

const ForecastChart = () => {
  const [forecastData, setForecastData] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [periods, setPeriods] = useState(12);

  const loadData = async () => {
    setLoading(true);
    try {
      const [forecastResponse, historicalResponse] = await Promise.all([
        getForecast(periods),
        getHistoricalData()
      ]);
      
      setForecastData(forecastResponse.forecast);
      setHistoricalData(historicalResponse.historical_data);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [periods]);

  // Combine historical and forecast data for the chart
  const chartData = [
    ...historicalData.map(item => ({
      date: item.date,
      sales: item.sales,
      type: 'Historical'
    })),
    ...forecastData.map(item => ({
      date: item.date,
      sales: item.predicted_sales,
      lower: item.prediction_lower,
      upper: item.prediction_upper,
      type: 'Forecast'
    }))
  ];

  return (
    <div className="forecast-chart">
      <h2>Sales Forecast</h2>
      
      <div className="chart-controls">
        <div className="form-group">
          <label>Forecast Periods (months):</label>
          <select 
            value={periods} 
            onChange={(e) => setPeriods(parseInt(e.target.value))}
            disabled={loading}
          >
            <option value={6}>6 months</option>
            <option value={12}>12 months</option>
            <option value={24}>24 months</option>
          </select>
        </div>
        
        <button onClick={loadData} disabled={loading}>
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {loading ? (
        <div className="loading">Loading chart data...</div>
      ) : (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                angle={-45}
                textAnchor="end"
                height={80}
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                tickFormatter={(value) => `$${value.toLocaleString()}`}
              />
              <Tooltip 
                formatter={(value) => [`$${Number(value).toLocaleString()}`, 'Sales']}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="sales" 
                stroke="#8884d8"
                name="Sales"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="forecast-details">
        <h3>Forecast Details</h3>
        <div className="forecast-table">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Predicted Sales</th>
                <th>Lower Bound</th>
                <th>Upper Bound</th>
              </tr>
            </thead>
            <tbody>
              {forecastData.map((item, index) => (
                <tr key={index}>
                  <td>{item.date}</td>
                  <td>${item.predicted_sales.toFixed(2)}</td>
                  <td>${item.prediction_lower.toFixed(2)}</td>
                  <td>${item.prediction_upper.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ForecastChart;