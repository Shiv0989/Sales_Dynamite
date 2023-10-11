import sys
from fastapi import FastAPI, Query
from joblib import load
import pandas as pd
import numpy as np

from src.models.forecasting_model import ForecastingModel
from src.models.predict_model import PredictiveModeling

app = FastAPI()
model_instance = PredictiveModeling()

forecasting_model_1 = load('models/forecasting/f_model.joblib')
predict_model = load('models/predictive/regression_model.joblib')

@app.get("/")
def read_root():
    """
    Endpoint that provides a brief overview of the Sales Forecasting API.
    
    Returns:
        dict: A dictionary containing a short description, list of available endpoints, and the GitHub repository link.
    """
    return {
        "description": "Welcome to the Sales Forecasting API! This API is designed to provide forecasts on national sales and predictions for individual store items.",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Provides a brief overview of the Sales Forecasting API including available endpoints."
            },
            {
                "path": "/health/",
                "method": "GET",
                "description": "API health check. Confirms that the API is up and running."
            },
            {
                "path": "/sales/national/",
                "method": "GET",
                "description": "Fetches a national sales forecast for the upcoming 7 days."
            },
            {
                "path": "/sales/stores/items/",
                "method": "GET",
                "description": "Predicts sales for a specific item in a particular store on a given date."
            }
        ],
        "github_repo": "https://github.com/Shiv0989/Sales_Dynamite.git"
    }

@app.get("/health/")
def health_check():
    """
    Health check endpoint for the Sales Forecasting API.
    
    Returns:
        dict: A dictionary containing a status code and a welcome message.
    """
    return {"status": 200, "message": "Sales Forecasting API is up and running!"}

@app.get("/sales/national/")
def national_sales_forecast(start_date: str = Query(..., description="The start date for the 7-day forecast. Format: YYYY-MM-DD")):
    forecast = forecasting_model_1.predict(start_date=start_date)
    return {"forecast": forecast}

@app.get("/sales/stores/items/")
def predict_sales(item_id: str, store_id: str, date: str):
    # Directly predicting using the model instance
    predicted_value = model_instance.predict_revenue(item_id, store_id, date)
    return {"prediction": float(predicted_value)}  # Convert numpy.float32 to float before sending
