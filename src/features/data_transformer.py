from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import os

os.chdir("/Users/shivatmaksharma/D/Uni Work/Advanced ML/Assignment 2/SalesDynamite")

class DataTransformerDebug(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        # Define a dictionary to store encoders
        self.encoders = {}

    def fit_and_save_encoders(self, data, columns):
        for col in columns:
            le = LabelEncoder().fit(data[col])
            self.encoders[col] = le
            # Save the encoder using joblib
            joblib.dump(le, f"models/{col}_encoder.joblib")
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert 'day' column (e.g., d_1) to integer format and then to datetime format
        X['d'] = X['d'].str.split('_').str[1].astype(int)
        start_date = '2011-01-29'  # Adjust this based on your dataset's start date
        base_date = pd.Timestamp(start_date)
        date_offsets = pd.to_timedelta(X['d'] - 1, unit='D')
        X['date'] = base_date + date_offsets


        # Extract year, month, day
        X['year'] = X['date'].dt.year
        X['month'] = X['date'].dt.month
        X['day_of_month'] = X['date'].dt.day
        
        # Convert event_name and event_type columns to category type
        X['event_name'] = X['event_name'].astype('category')
        X['event_type'] = X['event_type'].astype('category')
        
        # Label encode categorical columns or convert them to category codes
        categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name', 'event_type']
        
        # Train and save the encoders
        self.fit_and_save_encoders(X, categorical_cols)
        
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes

        # Drop unnecessary columns, including 'sales' and 'sell_price'
        X = X.drop(columns=['id', 'd', 'date', 'sales', 'sell_price', 'wm_yr_wk'])
        
        # Diagnostic print
        print("\nData types after DataTransformer:")
        print(X.dtypes)
        print(X.columns)
        
        # Uncomment if you want to save transformed data
        # X.to_csv("/Users/shivatmaksharma/D/Uni Work/Advanced ML/Assignment 2/SalesDynamite/data/processed/transformed_data.csv")
        
        return X
