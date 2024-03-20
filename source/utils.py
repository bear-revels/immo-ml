import geopandas as gpd
import joblib 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
from scipy import stats
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

def join_data(raw_data):
    """
    Join the raw data with external datasets.

    Parameters:
    raw_data (DataFrame): Raw data.

    Returns:
    DataFrame: Joined data.
    """
    # Load external datasets
    geo_data = pd.DataFrame(raw_data)
    pop_density_data = pd.read_excel('data/external_data/PopDensity.xlsx', dtype={'Refnis': int})
    house_income_data = pd.read_excel('data/external_data/HouseholdIncome.xlsx', dtype={'Refnis': int})
    property_value_data = pd.read_excel('data/external_data/PropertyValue.xlsx', dtype={'Refnis': int})

    # Define a function to create Point objects from latitude and longitude
    def create_point(row):
        try:
            latitude = float(row['Latitude'])
            longitude = float(row['Longitude'])
            return Point(longitude, latitude)
        except ValueError:
            return None

    # Create Point geometries from latitude and longitude coordinates in real estate data
    geo_data['geometry'] = geo_data.apply(create_point, axis=1)

    # Load the raw data into a GeoDataFrame
    geo_data = gpd.GeoDataFrame(raw_data, geometry=geo_data['geometry'], crs='EPSG:4326')

    # Read only the necessary column 'cd_munty_refnis' from the municipality GeoJSON file
    municipality_gdf = gpd.read_file('data/external_data/REFNIS_CODES.geojson', driver='GeoJSON')[['cd_munty_refnis', 'geometry']].to_crs(epsg=4326)

    # Perform spatial join with municipality data
    joined_data = gpd.sjoin(geo_data, municipality_gdf, how='left', predicate='within')

    # Convert 'cd_munty_refnis' column to int type
    joined_data['cd_munty_refnis'] = joined_data['cd_munty_refnis'].fillna(-1).astype(int)

    # Data Merge
    datasets = [pop_density_data, property_value_data, house_income_data]
    for dataset in datasets:
        joined_data = joined_data.merge(dataset, left_on='cd_munty_refnis', right_on='Refnis', how='left')
        joined_data.drop(columns=['Refnis'], inplace=True)

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

    # Task 1: Drop rows with empty values in 'Price', 'LivingArea', or 'BedroomCount' columns,
    # and drop rows where any of these columns contain infinite values
    cleaned_data = cleaned_data.dropna(subset=['Price', 'LivingArea', 'Longitude', 'Latitude'])
    cleaned_data = cleaned_data[~cleaned_data[['Price', 'LivingArea']].isin([np.inf, -np.inf]).any(axis=1)]

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

    # Add 1 to the BedroomCount column and fill null values with 1
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

    cleaned_data['Condition#'] = cleaned_data['Condition'].map(condition_mapping)
    cleaned_data['KitchenType#'] = cleaned_data['KitchenType'].map(kitchen_mapping)

    # Task 16: Remove specified columns
    columns_to_drop = [
    'ID', 'Street', 'HouseNumber', 'ViewCount', 'ConstructionYear', 'Box', 'City', 'Region', 'District', 'PropertyType',
    'SaleType', 'BidStylePricing', 'KitchenType', 'EPCScore', 'Terrace', 'Garden', 'Floor',
    'Condition', 'ListingExpirationDate', 'ListingCloseDate', 'Latitude', 'Longitude',
    'PropertyUrl', 'ListingCreateDate', 'Property url', 'geometry', 'bookmarkCount', 'index_right', 'cd_munty_refnis'
    ]

    cleaned_data.drop(columns=columns_to_drop, inplace=True)

    # Return the cleaned DataFrame
    return cleaned_data

def transform_features(cleaned_data):
    transformed_data = cleaned_data.copy()

    transformed_data['Price'] = np.log10(transformed_data['Price'] + 1)
    transformed_data['LivingArea'] = np.log10(transformed_data['LivingArea'] + 1)
    transformed_data['BedroomCount'] = np.log10(transformed_data['BedroomCount'] + 1)
    transformed_data['GardenArea'] = np.log10(transformed_data['GardenArea'] + 1)

    # Return the cleaned DataFrame
    return transformed_data

def engineer_features(transformed_data):
    engineered_data = transformed_data.copy()

    # Create a new column called PricePerSqm
    engineered_data['PricePerSqm'] = engineered_data['Price'] / engineered_data['LivingArea']

    # Create a new column called SqmPerBedroom
    engineered_data['SqmPerBedroom'] = engineered_data['LivingArea'] / engineered_data['BedroomCount']

    # Calculate z-scores within each group defined by 'PostalCode' and 'PropertySubType'
    z_scores = engineered_data.groupby(['PostalCode', 'PropertySubType'])[['PricePerSqm', 'SqmPerBedroom']].transform(stats.zscore)

    # Filter out rows where the absolute z-score is less than 3 for both columns
    engineered_data = engineered_data[(abs(z_scores['PricePerSqm']) < 3) & (abs(z_scores['SqmPerBedroom']) < 3)]

    # Drop columns if they exist
    columns_to_drop = ['PostalCode', 'PricePerSqm', 'SqmPerBedroom', 'Population']
    engineered_data = engineered_data.drop(columns=[col for col in columns_to_drop if col in engineered_data.columns], errors='ignore')

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
    # Encode categorical data using one-hot encoding
    encoded_data = pd.get_dummies(data, columns=['Province', 'PropertySubType'], dummy_na=False, drop_first=True)

    # Convert boolean columns to integer (0 or 1)
    bool_columns = encoded_data.select_dtypes(include=bool).columns
    encoded_data[bool_columns] = encoded_data[bool_columns].astype(int)

    return encoded_data

def impute_data(encoded_data):
    """
    Impute missing values in the DataFrame using SimpleImputer with median strategy.

    Parameters:
    encoded_data (DataFrame): The DataFrame with encoded categorical features

    Returns:
    DataFrame: The DataFrame with imputed missing values
    """
    # Create a copy of the encoded data
    imputed_data = encoded_data.copy()

    # Identify columns with missing values
    columns_with_missing_values = imputed_data.columns[imputed_data.isnull().any()].tolist()

    # Use SimpleImputer with median strategy to impute missing values
    imputer = SimpleImputer(strategy='median')
    imputed_data[columns_with_missing_values] = imputer.fit_transform(imputed_data[columns_with_missing_values])

    return imputed_data

def standardize_data(data):
    """
    Standardize the DataFrame by scaling numeric features.

    Parameters:
    data (DataFrame): The DataFrame to be standardized

    Returns:
    DataFrame: The standardized DataFrame
    """
    # Standardize the data using StandardScaler
    scaler = StandardScaler()
    standardized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return standardized_data

def execute_model(model_type, refresh_data):
    """
    Execute the specified model.

    Parameters:
    model_type (str): The type of model to execute
    refresh_data (bool): Whether to refresh the data.

    Returns:
    tuple: A tuple containing evaluation metrics (dict), actual values (array), and predicted values (array)
    """
    if model_type == "linear_regression":
        from source.models import execute_linear_regression
        metrics, y_test, y_pred = execute_linear_regression(refresh_data)
    elif model_type == "gradient_boosted_decision_tree":
        from source.models import execute_gradient_boosted_decision_tree
        metrics, y_test, y_pred = execute_gradient_boosted_decision_tree(refresh_data)
    elif model_type == "random_forest":
        from source.models import execute_random_forest
        metrics, y_test, y_pred = execute_random_forest(refresh_data)
    else:
        raise ValueError("Invalid model type. Please select a valid model.")

    return metrics, y_test, y_pred

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

def visualize_metrics(metrics, y_test, y_pred, comments=""):
    """
    Print the evaluation metrics of a model and visualize the predicted vs actual values.

    Parameters:
    metrics (dict): Dictionary containing evaluation metrics
    y_test (array-like): True target values
    y_pred (array-like): Predicted target values
    """
    # Extract metric values
    mse = metrics.get("Mean Squared Error")
    r_squared = metrics.get("R-squared value")

    # Convert R-squared value to percentage
    r_squared_percent = round(r_squared * 100, 2)

    # Format Mean Squared Error with commas and two decimal places
    formatted_mse = "{:,.2f}".format(mse)

    # Print the metrics
    print("Evaluation Metrics:")
    print("Mean Squared Error:", formatted_mse)
    print("R-squared value:", f"{r_squared_percent:.2f}%")

    # Plot the metrics
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Mean Squared Error
    ax[0].bar(["Mean Squared Error"], [mse], color='blue')
    ax[0].set_title(f"Mean Squared Error: {formatted_mse}")

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

    if comments:
        plt.text(0.5, -0.1, f"Comments: {comments}", horizontalalignment='center', verticalalignment='center', transform=ax[2].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()