import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const predictSales = async (date) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/predict`, { date });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Prediction failed');
  }
};

export const getForecast = async (periods = 12) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/forecast`, { periods, freq: 'M' });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to get forecast');
  }
};

export const getHistoricalData = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/historical-data`);
  return response.data;
};

export const getModelInfo = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/model-info`);
  return response.data;
};

export const healthCheck = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/health`);
  return response.data;
};