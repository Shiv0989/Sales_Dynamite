from prophet import Prophet
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from joblib import load
import os

os.chdir("/Users/shivatmaksharma/D/Uni Work/Advanced ML/Assignment 2/SalesDynamite")

class ForecastingModel(BaseEstimator, TransformerMixin):
    def __init__(self, periods=7):
        self.periods = periods  # Number of days to forecast
        self.model = load('models/forecasting/forecasting_model.joblib') #Prophet(daily_seasonality=True, yearly_seasonality=True)
        
    def fit(self, X, y=None):
        df = X.copy()
        
        # Convert 'event_type' to binary encoding: 0 for no event and 1 for any event
        df['event_type_binary'] = df['event_type'].apply(lambda x: 0 if pd.isnull(x) else 1)

        df = df.drop(columns=['event_type'])
        df = df.groupby('date').sum().reset_index()  
        
        df = df[['date', 'revenue', 'event_type_binary']]
        df.columns = ['ds', 'y', 'event_type_binary']
        
        # Inform Prophet about the additional regressor
        self.model.add_regressor('event_type_binary')
        
        # Store the columns for future use
        self.columns = df.columns
        
        self.model.fit(df)
        return self


    def predict(self, start_date=None, periods=None):
        # If no start_date is provided, forecast from the last date in the training data
        if start_date:
            last_date_in_data = pd.to_datetime(start_date)
        else:
            last_date_in_data = self.model.history_dates.max()

        # Use the provided periods or default to self.periods
        periods_to_forecast = periods or self.periods
            
        # Create a range of dates starting from the day after the last_date_in_data
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=periods_to_forecast)
            
        # Convert this to a DataFrame and merge with the existing future DataFrame
        future = pd.DataFrame({'ds': future_dates})
        
        # Add a placeholder for the 'event_type_binary' column
        future['event_type_binary'] = 0  # Adjust this as needed
        
        # Forecast
        forecast = self.model.predict(future)
            
        # Return only the forecasted values for the next 'periods' days
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_to_forecast)

 
    def evaluate(self, test, test_length):
        # Predict using the model
        forecast = self.predict(start_date=test['date'].iloc[0], periods=test_length)
        predicted_values = forecast['yhat'].values
        
        # Compute evaluation metrics
        mae = mean_absolute_error(test['revenue'].values, predicted_values)
        mse = mean_squared_error(test['revenue'].values, predicted_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test['revenue'].values - predicted_values) / test['revenue'].values)) * 100
        
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}

    
    def transform(self, X):
        # This can remain a no-op for this module as we're not transforming the input data
        return X
