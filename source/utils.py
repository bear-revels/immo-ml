import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

class FilterRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if 'Price' in X.columns and 'LivingArea' in X.columns and 'SaleType' in X.columns and 'BidStylePricing' in X.columns:
            X_filtered = X[(X['Price'].notnull()) & (X['LivingArea'].notnull()) & 
                           (X['SaleType'] == 'residential_sale') & (X['BidStylePricing'] != 1)]
            return X_filtered
        else:
            return X

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
        if 'PostalCode' in geo_data.columns:
            joined_data = geo_data.merge(self.postal_refnis[['PostalCode', 'Refnis']], 
                                    left_on='PostalCode', 
                                    right_on='PostalCode', 
                                    how='left')

        # Data Merge
        datasets = [self.pop_density_data, self.property_value_data, self.house_income_data]
        for dataset in datasets:
            if 'Refnis' in joined_data.columns:
                joined_data = joined_data.merge(dataset, left_on='Refnis', right_on='Refnis', how='left')

        # Return the resulting DataFrame
        return joined_data
    
from sklearn.base import BaseEstimator, TransformerMixin

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
        if isinstance(X, pd.DataFrame):
            columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
            X_dropped = X.drop(columns=columns_to_drop, errors='ignore')
            return X_dropped
        else:
            return X

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

# Define a preprocessing pipeline with ColumnTransformer
preprocessing = Pipeline(steps=[
    ('filter_rows', FilterRows()),  # Custom transformer to filter rows
    ('replace_nulls', ReplaceNulls()),  # Custom transformer to replace null values
    ('join_data', JoinData()),  # Custom transformer to join external datasets
    ('drop_columns', DropColumns()),  # Pass the columns to drop
    ('encode_categorical', EncodeCategorical()),  # Custom transformer to encode categorical variables
    ('feature_transformer', FeatureTransformer())])

def import_data(refresh=False):
    """
    Import data either from a CSV file or from an online source.

    Parameters:
    refresh (bool): Whether to refresh the data by downloading it again

    Returns:
    DataFrame: The imported raw data
    """
    if refresh:
        print("Loading and preprocessing new data...")
        raw_data = pd.read_csv("https://raw.githubusercontent.com/bear-revels/immo-eliza-scraping-Python_Pricers/main/data/all_property_details.csv", dtype={'PostalCode': str})
        raw_data.to_csv('./files/data/raw_data.csv', index=False, encoding='utf-8')

    else:
        print("Preprocessing the existing data...")
        raw_data = pd.read_csv('./files/data/raw_data.csv')
    return raw_data

def visualize_metrics(metrics, y_test, y_pred):
    """
    Print the evaluation metrics of a model and visualize the predicted vs actual values.

    Parameters:
    metrics (dict): Dictionary containing evaluation metrics
    y_test (array-like): True target values
    y_pred (array-like): Predicted target values
    """
    # Extract metric values
    mae = metrics.get("Mean Absolute Error")
    r_squared = metrics.get("R-squared value")

    # Convert R-squared value to percentage
    r_squared_percent = round(r_squared * 100, 2)

    # Format Mean Squared Error with commas and two decimal places
    formatted_mae = "{:,.2f}".format(mae)

    # Print the metrics
    print("Evaluation Metrics:")
    print("Mean Absolute Error:", formatted_mae)
    print("R-squared value:", f"{r_squared_percent:.2f}%")

    # Plot the metrics
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Mean Squared Error
    ax[0].bar(["Mean Absolute Error"], [mae], color='blue')
    ax[0].set_title(f"Mean Absolute Error: {formatted_mae}")

    # Plot R-squared value
    ax[1].bar(["R-squared value"], [r_squared_percent], color='green')
    ax[1].set_title(f"R-squared value: {r_squared_percent:.2f}%")
    ax[1].set_ylim([0, 100])  # Set y-axis limits to 0 and 100

    # Plot predicted vs actual values
    ax[2].scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Plot diagonal line
    ax[2].set_xlabel('Actual')
    ax[2].set_ylabel('Predicted')
    ax[2].set_title('Predicted vs Actual')

    plt.tight_layout()
    plt.show()

def save_model(model, filename, params=None, preprocessing_pipeline=None):
    """
    Save a machine learning model to a file.

    Parameters:
    model: The machine learning model to be saved
    filename (str): The name of the file to save the model to
    params (dict): Hyperparameters of the model
    preprocessing_pipeline: Preprocessing pipeline used on the data
    """
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # Save the model, parameters, and preprocessing pipeline to a dictionary
    model_data = {
        "model": model,
        "params": params,
        "preprocessing_pipeline": preprocessing_pipeline
    }

    filepath = os.path.join("./models", filename + ".pkl")
    joblib.dump(model_data, filepath)
    
    print(f"Model saved as {filepath}")