import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import SalesPredictor from './components/SalesPredictor';
import ModelInfo from './components/ModelInfo';
import ForecastChart from './components/ForecastChart';
import './styles/App.css';

function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    // Fetch model info on app start
    fetch('/api/model-info')
      .then(response => response.json())
      .then(data => setModelInfo(data))
      .catch(error => console.error('Error fetching model info:', error));
  }, []);

  return (
    <div className="App">
      <Header />
      <div className="tab-navigation">
        <button 
          className={activeTab === 'predict' ? 'active' : ''} 
          onClick={() => setActiveTab('predict')}
        >
          Sales Prediction
        </button>
        <button 
          className={activeTab === 'forecast' ? 'active' : ''} 
          onClick={() => setActiveTab('forecast')}
        >
          Forecast Chart
        </button>
        <button 
          className={activeTab === 'info' ? 'active' : ''} 
          onClick={() => setActiveTab('info')}
        >
          Model Info
        </button>
      </div>
      
      <div className="main-content">
        {activeTab === 'predict' && <SalesPredictor />}
        {activeTab === 'forecast' && <ForecastChart />}
        {activeTab === 'info' && <ModelInfo modelInfo={modelInfo} />}
      </div>
    </div>
  );
}

export default App;