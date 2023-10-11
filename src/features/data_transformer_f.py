from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DataTransformerDebug(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert 'day' column (e.g., d_1) to integer format and then to datetime format
        X['d'] = X['d'].str.split('_').str[1].astype(int)
        start_date = '2011-01-29'  # Adjust this based on your dataset's start date
        X['date'] = pd.to_datetime(X['d'].apply(lambda x: pd.Timestamp(start_date) + pd.DateOffset(days=x-1)))

        # Extract year, month, day
        X['year'] = X['date'].dt.year
        X['month'] = X['date'].dt.month
        X['day_of_month'] = X['date'].dt.day
        
        # Convert event_name and event_type columns to category type
        X['event_name'] = X['event_name'].astype('category')
        X['event_type'] = X['event_type'].astype('category')
        
        # Label encode categorical columns or convert them to category codes
        categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name', 'event_type']
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes

        # Drop unnecessary columns, including 'sales' and 'sell_price'
        X = X.drop(columns='id')
        
        # Diagnostic print
        print("\nData types after DataTransformer:")
        print(X.dtypes)
        
        return X
