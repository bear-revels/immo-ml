from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import numpy as np
import os
import pandas as pd

def clean_data(raw_data):
    """
    Clean the raw data by performing several tasks.

    Parameters:
    raw_data (DataFrame): The raw DataFrame to be cleaned
    """

    cleaned_data = raw_data.copy()

    # Task 1: Drop rows with empty values in 'Price', 'LivingArea', or 'BedroomCount' columns,
    # and drop rows where any of these columns contain infinite values
    cleaned_data = cleaned_data.dropna(subset=['Price', 'LivingArea', 'BedroomCount'])
    cleaned_data = cleaned_data[~cleaned_data[['Price', 'LivingArea', 'BedroomCount']].isin([np.inf, -np.inf]).any(axis=1)]

    # Task 2: Remove duplicates in the 'ID' column and where all columns but 'ID' are equal
    cleaned_data.drop_duplicates(subset='ID', inplace=True)
    cleaned_data.drop_duplicates(subset=cleaned_data.columns.difference(['ID']), keep='first', inplace=True)

    # Task 3: Convert empty values to 0 for specified columns; assumption that if blank then 0
    columns_to_fill_with_zero = ['Furnished', 'Fireplace', 'TerraceArea', 'GardenArea', 'SwimmingPool', 'BidStylePricing', 'ViewCount', 'bookmarkCount']
    cleaned_data[columns_to_fill_with_zero] = cleaned_data[columns_to_fill_with_zero].fillna(0)

    # Task 4: Filter rows where SaleType == 'residential_sale' and BidStylePricing == 0
    cleaned_data = cleaned_data[(cleaned_data['SaleType'] == 'residential_sale') & (cleaned_data['BidStylePricing'] == 0)].copy()

    # Task 5: Adjust text format
    columns_to_str = ['PropertySubType', 'KitchenType', 'Condition', 'EPCScore']

    def adjust_text_format(x):
        if isinstance(x, str):
            return x.title()
        else:
            return x

    cleaned_data.loc[:, columns_to_str] = cleaned_data.loc[:, columns_to_str].map(adjust_text_format)

    # Task 6: Remove leading and trailing spaces from string columns
    cleaned_data.loc[:, columns_to_str] = cleaned_data.loc[:, columns_to_str].apply(lambda x: x.str.strip() if isinstance(x, str) else x)

    # Task 7: Replace the symbol '�' with 'e' in all string columns
    cleaned_data = cleaned_data.map(lambda x: x.replace('�', 'e') if isinstance(x, str) else x)

    # Task 8: Fill missing values with None and convert specified columns to float64 type
    columns_to_fill_with_none = ['EnergyConsumptionPerSqm']
    cleaned_data[columns_to_fill_with_none] = cleaned_data[columns_to_fill_with_none].where(cleaned_data[columns_to_fill_with_none].notna(), None)

    columns_to_float64 = ['TerraceArea', 'GardenArea', 'EnergyConsumptionPerSqm']
    cleaned_data[columns_to_float64] = cleaned_data[columns_to_float64].astype(float)

    # Task 9: Convert specified columns to Int64 type
    columns_to_int64 = ['PostalCode', 'ConstructionYear', 'Floor', 'Furnished', 'Fireplace','Facades', 'SwimmingPool', 'bookmarkCount', 'ViewCount']
    cleaned_data[columns_to_int64] = cleaned_data[columns_to_int64].astype(float).round().astype('Int64')

    # Task 10: Replace any ConstructionYear > current_year + 10 with None
    current_year = datetime.now().year
    max_construction_year = current_year + 10
    cleaned_data['ConstructionYear'] = cleaned_data['ConstructionYear'].where(cleaned_data['ConstructionYear'] <= max_construction_year, None)

    # Task 11: Trim text after and including '_' from the 'EPCScore' column
    cleaned_data['EPCScore'] = cleaned_data['EPCScore'].str.split('_').str[0]

    # Task 12: Convert 'ListingCreateDate' to integer timestamp
    cleaned_data['ListingCreateDate'] = cleaned_data['ListingCreateDate'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()))

    # Task 13: Replace values less than or equal to 0 in 'EnergyConsumptionPerSqm' with 0
    cleaned_data.loc[cleaned_data['EnergyConsumptionPerSqm'] < 0, 'EnergyConsumptionPerSqm'] = 0

    # Add 1 to the BedroomCount column to avoid division by zero later
    cleaned_data['BedroomCount'] += 1

    # Task 15: Convert string values to numeric values using dictionaries for specified columns
    condition_mapping = {
        'nan': None,
        'To_Be_Done_Up': 2,
        'To_Renovate': 1,
        'Just_Renovated': 4,
        'As_New': 5,
        'Good': 3,
        'To_Restore': 0
    }

    kitchen_mapping = {
        'nan': None,
        'Installed': 1,
        'Not_Installed': 0,
        'Hyper_Equipped': 1,
        'Semi_Equipped': 1,
        'Usa_Installed': 1,
        'Usa_Hyper_Equipped': 1,
        'Usa_Semi_Equipped': 1,
        'Usa_Uninstalled': 0
    }

    cleaned_data['Condition#'] = cleaned_data['Condition'].map(condition_mapping)
    cleaned_data['KitchenType#'] = cleaned_data['KitchenType'].map(kitchen_mapping)

    # Task 16: Remove specified columns
    columns_to_drop = ['ID', 
                       'Street', 
                       'HouseNumber', 
                       'Box',
                       'City',
                       'Region', 
                       'District', 
                       'Province', 
                       'PropertyType', 
                       'SaleType', 
                       'BedroomCount',
                       'BidStylePricing', 
                       'BedroomCount', 
                       'KitchenType',
                       'Terrace', 
                       'Garden', 
                       'EPCScore',
                       'Condition',
                       'Latitude', 
                       'Longitude', 
                       'ListingExpirationDate', 
                       'ListingCloseDate', 
                       'PropertyUrl', 
                       'Property url']
    cleaned_data.drop(columns=columns_to_drop, inplace=True)

    # Task 17: Convert 'PropertySubType' categorical data into numeric features using one-hot encoding
    cleaned_data = pd.get_dummies(cleaned_data, columns=['PropertySubType'], prefix='Property', dummy_na=False, drop_first=True)

    # Save the cleaned data to a CSV file
    cleaned_data.to_csv('./data/clean_data.csv', index=False, encoding='utf-8')

    # Return the cleaned DataFrame
    return cleaned_data

def normalize_data(cleaned_data):
    """
    Normalize the cleaned data by scaling numeric features.

    Parameters:
    cleaned_data (DataFrame): The cleaned DataFrame to be normalized

    Returns:
    DataFrame: The normalized DataFrame
    """
    normal_data = cleaned_data.copy()

    # Task 1: Impute null values for 'Floor'
    floor_imputer = SimpleImputer(strategy='median')
    normal_data['Floor'] = floor_imputer.fit_transform(normal_data[['Floor']])

    # Task 2: Impute null values for 'ConstructionYear'
    construction_year_imputer = SimpleImputer(strategy='median')
    normal_data['ConstructionYear'] = construction_year_imputer.fit_transform(normal_data[['ConstructionYear']])

    # Task 3: Impute null values for 'Facades'
    facades_imputer = SimpleImputer(strategy='median')
    normal_data['Facades'] = facades_imputer.fit_transform(normal_data[['Facades']])

    # Task 4: Impute null values for 'EnergyConsumptionPerSqm'
    energy_imputer = SimpleImputer(strategy='median')
    normal_data['EnergyConsumptionPerSqm'] = energy_imputer.fit_transform(normal_data[['EnergyConsumptionPerSqm']])

    # Task 5: Impute null values for 'Condition#'
    condition_imputer = SimpleImputer(strategy='median')
    normal_data['Condition#'] = condition_imputer.fit_transform(normal_data[['Condition#']])

    # Task 6: Impute null values for 'KitchenType#'
    kitchen_imputer = SimpleImputer(strategy='median')
    normal_data['KitchenType#'] = kitchen_imputer.fit_transform(normal_data[['KitchenType#']])

    # Task 7: Round float columns to the nearest whole number and convert to int64 type
    float_columns = normal_data.select_dtypes(include=['float64']).columns
    normal_data[float_columns] = normal_data[float_columns].round().astype('Int64')
    
    normal_data = normal_data.astype(int)  # Convert boolean values to integers (1s and 0s)

    # Select the columns to normalize
    columns_to_normalize = ['Floor', 'PostalCode', 'Price', 'ConstructionYear', 'Furnished', 'Fireplace', 'TerraceArea', 'GardenArea', 'Facades', 'SwimmingPool', 'EnergyConsumptionPerSqm', 'ListingCreateDate', 'bookmarkCount', 'ViewCount', 'LivingArea', 'Condition#', 'KitchenType#']

    # Initialize the StandardScaler and MinMaxScaler
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    # Fit and transform the data with StandardScaler
    normal_data[columns_to_normalize] = scaler_standard.fit_transform(normal_data[columns_to_normalize])

    # Fit and transform the data with MinMaxScaler
    normal_data[columns_to_normalize] = scaler_minmax.fit_transform(normal_data[columns_to_normalize])

    return normal_data

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
        raw_data.to_csv('./data/raw_data.csv', index=False, encoding='utf-8')
    else:
        print("Preprocessing the existing data...")
        raw_data = pd.read_csv('./data/raw_data.csv')
    return raw_data

def load_model(filename):
    """
    Load a machine learning model from a file.

    Parameters:
    filename (str): The name of the file containing the model

    Returns:
    model: The loaded machine learning model
    """
    filepath = os.path.join("./models", filename + ".pkl")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def save_model(model, filename):
    """
    Save a machine learning model to a file.

    Parameters:
    model: The machine learning model to be saved
    filename (str): The name of the file to save the model to
    """
    if not os.path.exists("./models"):
        os.makedirs("./models")
    filepath = os.path.join("./models", filename + ".pkl")
    joblib.dump(model, filepath)
    print(f"Model saved as {filepath}")

def execute_model(selected_model, refresh=False):
    """
    Execute the selected machine learning model.

    Parameters:
    selected_model (str): The name of the selected model
    refresh (bool): Whether to refresh the data before execution
    """
    if selected_model == "linear_regression":
        from source.models import execute_linear_regression
        execute_linear_regression(refresh)
    elif selected_model == "auto_ml":
        from source.models import execute_auto_ml
        execute_auto_ml(refresh)
    elif selected_model == "multi_linear_regression":
        from source.models import execute_multi_linear_regression
        execute_multi_linear_regression(refresh)
    else:
        print("Selected model not found.")