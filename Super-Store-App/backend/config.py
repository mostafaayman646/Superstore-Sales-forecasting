import os

class Config:
    """Configuration settings for the Flask application"""
    
    # Model paths
    MODEL_PATH = 'Model/sales_model.json'
    METADATA_PATH = 'Model/model_metadata.json'
    TRAINING_DATA_PATH = 'Model/monthly_training_data.csv'
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    DEBUG = True
    
    # CORS settings
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']