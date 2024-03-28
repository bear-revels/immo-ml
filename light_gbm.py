import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

class FilterRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Your filtering logic here
        X_filtered = X[(X['Price'].notnull()) & (X['LivingArea'].notnull()) & 
                       (X['SaleType'] == 'residential_sale') & (X['BidStylePricing'] != 1)]
        return X_filtered

class ReplaceNulls(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_fill = ['Furnished', 'Fireplace', 'Terrace', 'TerraceArea', 'Garden', 'GardenArea', 'SwimmingPool', 'BidStylePricing', 'ViewCount', 'bookmarkCount']

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_filled = X.copy()
        for column in self.columns_to_fill:
            if column in X_filled.columns:
                X_filled[column] = X_filled[column].fillna(0)
        return X_filled
    
class JoinData(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Load external datasets
        self.postal_refnis = pd.read_excel('files/data/REFNIS_Mapping.xlsx', dtype={'Refnis': int})
        self.pop_density_data = pd.read_excel('files/data/PopDensity.xlsx', dtype={'Refnis': int})
        self.house_income_data = pd.read_excel('files/data/HouseholdIncome.xlsx', dtype={'Refnis': int})
        self.property_value_data = pd.read_excel('files/data/PropertyValue.xlsx', dtype={'Refnis': int})
        
    def fit(self, X, y=None):
        # Joining external datasets doesn't require fitting on any data,
        # so we simply return self.
        return self

    def transform(self, raw_data):
        if isinstance(raw_data, pd.DataFrame):
            geo_data = pd.DataFrame(raw_data)
        else:
            geo_data = pd.DataFrame.from_dict(raw_data, orient='index').T

        # Merge Refnis
        joined_data = geo_data.merge(self.postal_refnis[['PostalCode', 'Refnis']], 
                                left_on='PostalCode', 
                                right_on='PostalCode', 
                                how='left')

        # Data Merge
        datasets = [self.pop_density_data, self.property_value_data, self.house_income_data]
        for dataset in datasets:
            joined_data = joined_data.merge(dataset, left_on='Refnis', right_on='Refnis', how='left')

        # Return the resulting DataFrame
        return joined_data
    
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ['ID', 'Street', 'HouseNumber', 'Box', 'Floor', 'City', 
                           'SaleType', 'KitchenType', 'Latitude', 'Longitude', 
                           'ListingCreateDate', 'ListingExpirationDate', 
                           'ListingCloseDate', 'PropertyUrl', 'Property url',
                           'bookmarkCount', 'ViewCount', 'Refnis', 'BidStylePricing',
                           'ConstructionYear']

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Your column dropping logic here
        X_dropped = X.drop(columns=self.columns_to_drop)
        return X_dropped

class EncodeCategorical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_encoded = X.copy()

        condition_mapping = {
            'nan': None,
            'TO_BE_DONE_UP': 2,
            'TO_RENOVATE': 1,
            'JUST_RENOVATED': 4,
            'AS_NEW': 5,
            'GOOD': 3,
            'TO_RESTORE': 0
        }

        if 'Condition' in X_encoded.columns:
            X_encoded['Condition'] = X_encoded['Condition'].map(condition_mapping)

        if 'EPCScore' in X_encoded.columns:
            X_encoded['EPCScore'] = X_encoded['EPCScore'].str.split('_').str[0]

        for column in X_encoded.columns:
            if X_encoded[column].dtype == 'object':
                X_encoded[column] = LabelEncoder().fit_transform(X_encoded[column])
        return X_encoded

class FeatureTransformer:
    def __init__(self):
        self.columns_to_transform = ['Price', 'LivingArea', 'BedroomCount', 'GardenArea']

    def fit(self, X, y=None):
        # This method doesn't do anything for this transformer
        return self

    def transform(self, X):
        # Make a copy of the input DataFrame
        transformed_data = X.copy()

        # Apply transformation to specified columns
        for column in self.columns_to_transform:
            if column in transformed_data.columns and pd.api.types.is_numeric_dtype(transformed_data[column]):
                transformed_data[column] = np.log10((transformed_data[column] + 1))

        return transformed_data

def predict(raw_data_path):
    # Load raw data
    raw_data = pd.read_csv(raw_data_path)

    # Define a preprocessing pipeline with ColumnTransformer
    preprocessing = Pipeline(steps=[
        ('filter_rows', FilterRows()),  # Custom transformer to filter rows
        ('replace_nulls', ReplaceNulls()),  # Custom transformer to replace null values
        ('join_data', JoinData()),  # Custom transformer to join external datasets
        ('drop_columns', DropColumns()),  # Pass the columns to drop
        ('encode_categorical', EncodeCategorical()),  # Custom transformer to encode categorical variables
        ('feature_transformer', FeatureTransformer())
    ])

    # Define the full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing),
        ('model', LGBMRegressor())
    ])

    # Apply preprocessing pipeline
    preprocessed_data = full_pipeline.named_steps['preprocessing'].fit_transform(raw_data)

    # Separate features and target variable
    X = preprocessed_data.drop(columns=['Price'])
    y = preprocessed_data['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
    'num_leaves': [30, 50, 100],
    'max_depth': [5, 10, -1],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'lambda_l1': [0, 0.1, 0.5],  # Equivalent to reg_alpha
    'lambda_l2': [0, 0.1, 0.5],  # Equivalent to reg_lambda
    'force_row_wise': [True],  # Optimize training process
    'force_col_wise': [False]   # Optimize training process
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(full_pipeline['model'], param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best estimator and its corresponding score
    best_pipeline = grid_search.best_estimator_

    # Fit the best model on the training data
    best_pipeline.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = best_pipeline.predict(X_test)

    # Apply inverse log transformation to y_test and y_pred
    y_pred = np.power(10, y_pred) - 1
    y_test = np.power(10, y_test) - 1

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r_squared)

    # Save the best model
    joblib.dump(best_pipeline, 'models/LightGBM_Best_Pipeline.pkl')

    # Print feature importance
    feature_importance = best_pipeline.feature_importances_
    print("Feature Importance:", feature_importance)

# Call the predict function with the path to the raw data file
predict('files/data/raw_data.csv')