from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
import joblib 
from joblib import load
import numpy as np
import os

os.chdir("/Users/shivatmaksharma/D/Uni Work/Advanced ML/Assignment 2/SalesDynamite")

class PredictiveModeling(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='revenue'):
        self.target_column = target_column
        self.model = load('models/predictive/regression_model.joblib')

        
    '''def objective(self, space):
        model = XGBRegressor(
            n_estimators = 835,#int(space['n_estimators']),
            max_depth = 9,#int(space['max_depth']),
            learning_rate = 0.12805117452107706,#space['learning_rate'],
            gamma = 0.6496510747545003,#space['gamma'],
            min_child_weight = 8,#space['min_child_weight'],
            subsample = 0.10551259308117146,#space['subsample'],
            colsample_bytree = 0.99936767565064#space['colsample_bytree']
        )
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_train)
        
        # Compute metrics
        mse = mean_squared_error(self.y_train, predictions)
        rmse = mean_squared_error(self.y_train, predictions, squared=False)
        mae = mean_absolute_error(self.y_train, predictions)
        r2 = r2_score(self.y_train, predictions)
        
        # Return the main metric for optimization (mse in this case) and any additional metrics
        return {
            'loss': mse,  # primary metric for optimization
            'status': STATUS_OK,
            'attachments': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }'''

    def fit(self, X, y=None):
        
        self.feature_data = X.copy()

        # Check if the target column exists before separating it out
        if self.target_column in X.columns:
            y = X[self.target_column]
            X = X.drop(columns=[self.target_column])

        '''# Define the space for hyperparameter optimization
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 1000, 1),
            'max_depth': hp.quniform('max_depth', 1, 13, 1),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'gamma': hp.uniform('gamma', 0, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1)
        }
        
        # To use in the objective function
        self.X_train = X
        self.y_train = y

        # Run the optimizer
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials)
        
        best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'gamma': best['gamma'],
        'min_child_weight': int(best['min_child_weight']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree']
        }
        print("Best hyperparameters:", best_params)'''

        # Train the model with the best hyperparameters
        self.model = XGBRegressor(
            n_estimators=835,
            max_depth=9,
            learning_rate=0.12805117452107706,
            gamma=0.6496510747545003,
            min_child_weight=8,
            subsample=0.10551259308117146,
            colsample_bytree=0.99936767565064
        )
        self.model.fit(X, y)

        predictions = self.model.predict(X)

        # Store the predicted revenue values
        self.predicted_revenue = predictions

        # Compute metrics
        self.mse = mean_squared_error(y, predictions)
        self.rmse = mean_squared_error(y, predictions, squared=False)
        self.mae = mean_absolute_error(y, predictions)
        self.r2 = r2_score(y, predictions)

        return self
    
    
    def get_metrics(self):
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2
        }

    def transform(self, X):
        return X

    def predict_revenue(self, item_id, store_id, date):
        # Create a DataFrame from the provided data
        prediction_df = pd.DataFrame({"item_id": [item_id], "store_id": [store_id], "date": [date]})

        # Feature extraction
        prediction_df['date'] = pd.to_datetime(prediction_df['date'])
        prediction_df['cat_id'] = prediction_df['item_id'].str.split('_').str[0]
        prediction_df['state_id'] = prediction_df['store_id'].str.split('_').str[0]
        prediction_df['dept_id'] = prediction_df['item_id'].apply(lambda x: x.rsplit('_', 1)[0])
        prediction_df['year'] = prediction_df['date'].dt.year
        prediction_df['month'] = prediction_df['date'].dt.month
        prediction_df['day_of_month'] = prediction_df['date'].dt.day

        # Use mode of features in the training data to fill in for event details. 
        # This is a simplification and may need more sophisticated handling in real-world applications.
        prediction_df['event_name'] = 0
        prediction_df['event_type'] = 0

        # Use the previously saved encoders for encoding
        cat_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name', 'event_type']
        for col in cat_columns:
            le = joblib.load(f"/models/{col}_encoder.joblib")

            # Check if new category is in the training data
            prediction_df[col] = prediction_df[col].map(lambda s: s if s in le.classes_ else 'unseen_category')
            le.classes_ = np.append(le.classes_, 'unseen_category')  # Add the unseen category label

            prediction_df[col] = le.transform(prediction_df[col])

        prediction_df = prediction_df[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name', 'event_type', 'year', 'month', 'day_of_month']]

        # Predict using the trained model
        predicted_revenue = self.model.predict(prediction_df)
    
        return predicted_revenue[0]
    



