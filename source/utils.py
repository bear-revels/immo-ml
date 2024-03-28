import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

def join_data(raw_data):
    """
    Join the raw data with external datasets.

    Parameters:
    raw_data (DataFrame): Raw data.

    Returns:
    DataFrame: Joined data.
    """
    if isinstance(raw_data, pd.DataFrame):
        geo_data = pd.DataFrame(raw_data)
    else:
        # Convert input_data dictionary to DataFrame
        geo_data = pd.DataFrame.from_dict(raw_data, orient='index').T
    
    # Load external datasets
    postal_refnis = pd.read_excel('files/data/REFNIS_Mapping.xlsx', dtype={'Refnis': int})
    pop_density_data = pd.read_excel('files/data/PopDensity.xlsx', dtype={'Refnis': int})
    house_income_data = pd.read_excel('files/data/HouseholdIncome.xlsx', dtype={'Refnis': int})
    property_value_data = pd.read_excel('files/data/PropertyValue.xlsx', dtype={'Refnis': int})

    # Convert 'cd_munty_refnis' column to int type
    joined_data = geo_data.merge(postal_refnis[['PostalCode', 'Refnis']], 
                             left_on='PostalCode', 
                             right_on='PostalCode', 
                             how='left')

    # Data Merge
    datasets = [pop_density_data, property_value_data, house_income_data]
    for dataset in datasets:
        joined_data = joined_data.merge(dataset, left_on='Refnis', right_on='Refnis', how='left')

    # Return the resulting DataFrame
    return joined_data
        
def clean_data(raw_data):
    """
    Clean the raw data by performing several tasks.

    Parameters:
    raw_data (DataFrame): The raw DataFrame to be cleaned

    Returns:
    DataFrame: The cleaned DataFrame
    """

    cleaned_data = raw_data.copy()

    # Task 1: Drop rows with empty values in specified columns ('Price', 'LivingArea', 'Longitude', 'Latitude')
    columns_to_dropna = ['Price', 'LivingArea', 'PostalCode']
    for column in columns_to_dropna:
        if column in cleaned_data.columns:
            cleaned_data = cleaned_data.dropna(subset=[column])
            cleaned_data = cleaned_data[~cleaned_data[column].isin([np.inf, -np.inf])]

    # Task 2: Remove duplicates in the 'ID' column and where all columns but 'ID' are equal
    if 'ID' in cleaned_data.columns:
        cleaned_data.drop_duplicates(subset='ID', inplace=True)
        cleaned_data.drop_duplicates(subset=cleaned_data.columns.difference(['ID']), keep='first', inplace=True)

    # Task 3: Convert empty values to 0 for specified columns; assumption that if blank then 0
    columns_to_fill_with_zero = ['Furnished', 'Fireplace', 'TerraceArea', 'GardenArea', 'SwimmingPool', 'BidStylePricing', 'ViewCount', 'bookmarkCount']
    for column in columns_to_fill_with_zero:
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].fillna(0)

    # Task 4: Filter rows where SaleType == 'residential_sale' and BidStylePricing == 0
    if 'SaleType' in cleaned_data.columns and 'BidStylePricing' in cleaned_data.columns:
        cleaned_data = cleaned_data[(cleaned_data['SaleType'] == 'residential_sale') & (cleaned_data['BidStylePricing'] == 0)].copy()

    # Task 5: Adjust text format
    columns_to_str = ['PropertySubType', 'KitchenType', 'Condition', 'EPCScore']
    for column in columns_to_str:
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].apply(lambda x: x.title() if isinstance(x, str) else x)
            cleaned_data[column] = cleaned_data[column].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Task 8: Fill missing values with None and convert specified columns to float64 type
    columns_to_fill_with_none = ['EnergyConsumptionPerSqm']
    for column in columns_to_fill_with_none:
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].where(cleaned_data[column].notna(), None)

    columns_to_float64 = ['TerraceArea', 'GardenArea', 'EnergyConsumptionPerSqm']
    for column in columns_to_float64:
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].astype(float)

    # Task 9: Convert specified columns to Int64 type
    columns_to_int64 = ['PostalCode', 'ConstructionYear', 'Floor', 'Furnished', 'Fireplace','Facades', 'SwimmingPool', 'bookmarkCount', 'ViewCount']
    for column in columns_to_int64:
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].astype(float).round().astype('Int64')

    # Task 10: Replace any ConstructionYear > current_year + 10 with None
    if 'ConstructionYear' in cleaned_data.columns:
        current_year = datetime.now().year
        max_construction_year = current_year + 10
        cleaned_data['ConstructionYear'] = cleaned_data['ConstructionYear'].where(cleaned_data['ConstructionYear'] <= max_construction_year, None)

    # Task 11: Trim text after and including '_' from the 'EPCScore' column
    if 'EPCScore' in cleaned_data.columns:
        cleaned_data['EPCScore'] = cleaned_data['EPCScore'].str.split('_').str[0]

    # Task 12: Convert 'ListingCreateDate' to integer timestamp
    if 'ListingCreateDate' in cleaned_data.columns:
        cleaned_data['ListingCreateDate'] = cleaned_data['ListingCreateDate'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()))

    # Task 13: Replace values less than or equal to 0 in 'EnergyConsumptionPerSqm' with 0
    if 'EnergyConsumptionPerSqm' in cleaned_data.columns:
        cleaned_data.loc[cleaned_data['EnergyConsumptionPerSqm'] <= 0, 'EnergyConsumptionPerSqm'] = 0

    # Add 14 to the BedroomCount column and fill null values with 1
    if 'BedroomCount' in cleaned_data.columns:
        cleaned_data['BedroomCount'] = cleaned_data['BedroomCount'].fillna(0) + 1

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

    if 'Condition' in cleaned_data.columns:
        cleaned_data['Condition#'] = cleaned_data['Condition'].map(condition_mapping)

    if 'KitchenType' in cleaned_data.columns:
        cleaned_data['KitchenType#'] = cleaned_data['KitchenType'].map(kitchen_mapping)

    # Task 16: Remove specified columns
    columns_to_drop = [
        'ID', 'Street', 'HouseNumber', 'ViewCount', 'ConstructionYear', 'Box', 'City', 'Region', 'District', 'PropertyType',
        'SaleType', 'BidStylePricing', 'KitchenType', 'EPCScore', 'Terrace', 'Garden', 'Floor',
        'Condition', 'ListingExpirationDate', 'ListingCloseDate', 'Latitude', 'Longitude', 
        'PropertyUrl', 'ListingCreateDate', 'Property url', 'geometry', 'bookmarkCount', 
        'index_right', 'Refnis'
    ]

    for column in columns_to_drop:
        if column in cleaned_data.columns:
            cleaned_data.drop(columns=column, inplace=True)

    # Return the cleaned DataFrame
    return cleaned_data

def transform_features(cleaned_data):
    transformed_data = cleaned_data.copy()

    columns_to_transform = ['Price', 'LivingArea', 'BedroomCount', 'GardenArea']

    for column in columns_to_transform:
        if column in transformed_data.columns and pd.api.types.is_numeric_dtype(transformed_data[column]):
            transformed_data[column] = np.log10((transformed_data[column] + 1))

    # Return the transformed DataFrame
    return transformed_data

def engineer_features(transformed_data):
    engineered_data = transformed_data.copy()

    # Create a new column called PricePerSqm if 'Price' and 'LivingArea' columns are present
    if 'Price' in engineered_data.columns and 'LivingArea' in engineered_data.columns:
        engineered_data['PricePerSqm'] = engineered_data['Price'] / engineered_data['LivingArea']

    # Create a new column called SqmPerBedroom if 'LivingArea' and 'BedroomCount' columns are present
    if 'LivingArea' in engineered_data.columns and 'BedroomCount' in engineered_data.columns:
        engineered_data['SqmPerBedroom'] = engineered_data['LivingArea'] / engineered_data['BedroomCount']

    # Calculate z-scores within each group defined by 'PostalCode' and 'PropertySubType'
    if 'PricePerSqm' in engineered_data.columns and 'SqmPerBedroom' in engineered_data.columns:
        z_scores = engineered_data.groupby(['PostalCode', 'PropertySubType'])[['PricePerSqm', 'SqmPerBedroom']].transform(stats.zscore)

        # Filter out rows where the absolute z-score is less than 3 for both columns
        engineered_data = engineered_data[(abs(z_scores['PricePerSqm']) < 3) & (abs(z_scores['SqmPerBedroom']) < 3)]

    # Drop columns if they exist
    columns_to_drop = ['PricePerSqm', 'SqmPerBedroom', 'Population']
    engineered_data = engineered_data.drop(columns=[col for col in columns_to_drop if col in engineered_data.columns], errors='ignore')
    engineered_data['PostalCode'] = engineered_data['PostalCode'].astype(str)

    # Return the cleaned DataFrame
    return engineered_data

def encode_data(data):
    """
    Encode categorical data using one-hot encoding.

    Parameters:
    data (DataFrame): The DataFrame to be encoded

    Returns:
    DataFrame: The DataFrame with encoded categorical features
    """
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    return data

def standardize_data(data):
    """
    Standardize the DataFrame by scaling numeric features.

    Parameters:
    data (DataFrame): The DataFrame to be standardized

    Returns:
    DataFrame: The standardized DataFrame
    """
    # Check if there are numeric features to be standardized
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.empty:
        # Standardize the data using StandardScaler for numeric features
        scaler = StandardScaler()
        standardized_data = data.copy()
        standardized_data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        return standardized_data
    else:
        return data