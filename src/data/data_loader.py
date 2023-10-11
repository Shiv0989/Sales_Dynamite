import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataLoaderDebug(BaseEstimator, TransformerMixin):
    def __init__(self, filepath_train, filepath_cal, filepath_event, filepath_ippw):
        self.filepath_train = filepath_train
        self.filepath_cal = filepath_cal
        self.filepath_event = filepath_event 
        self.filepath_ippw = filepath_ippw
        
    def fit(self, X=None, y=None):
        # No-op
        return self

    def transform(self, X=None):
        # Load all datasets
        sales_train = pd.read_csv(self.filepath_train)
        calendar = pd.read_csv(self.filepath_cal)
        calendar_events = pd.read_csv(self.filepath_event)
        items_weekly_sell_prices = pd.read_csv(self.filepath_ippw)
        
        # Merge steps:
        # 1. Link sales to dates
        sales_train = sales_train.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], 
                                    var_name="d", value_name="sales")
        
        # Print data types after melt
        print("\nData after melt:\n", sales_train)
        
        sales_train = sales_train.merge(calendar, on="d", how="left")
        
        # Print data types after first merge
        print("\nData types after first merge:\n", sales_train.dtypes)
        
        # 2. Add event information
        sales_train = sales_train.merge(calendar_events, on="date", how="left")
        
        # Print data types after second merge
        print("\nData types after second merge:\n", sales_train.dtypes)
        
        # 3. Add sell prices
        sales_train = sales_train.merge(items_weekly_sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        
        # Compute the revenue column
        sales_train['revenue'] = sales_train['sell_price'] * sales_train['sales']

        # Fill NaN values in sell_price with 0
        sales_train['sell_price'].fillna(0, inplace=True)

        # Recompute the revenue after filling NaN values
        sales_train['revenue'] = sales_train['sell_price'] * sales_train['sales']

        print("\nData types after DataLoader:")
        print(sales_train.dtypes)
        
        return sales_train
    
