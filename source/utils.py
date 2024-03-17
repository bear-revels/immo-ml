import geopandas as gpd
import joblib 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
from scipy import stats
from shapely.geometry import Point
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
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
    geo_data = gpd.GeoDataFrame(raw_data)

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

    # Explicitly set CRS to WGS 84 (EPSG:4326)
    geo_data.crs = 'EPSG:4326'

    # Set geometry column after creating 'geometry' column
    geo_data = geo_data.set_geometry('geometry')  # Explicitly set the active geometry column
    
    # Explicitly set CRS to WGS 84 (EPSG:4326)
    geo_data.crs = 'EPSG:4326'

    municipality_gdf = gpd.read_file('./data/external_data/REFNIS_CODES.geojson', driver='GeoJSON').to_crs(epsg=4326)

    # Perform spatial join with municipality data
    joined_data = gpd.sjoin(geo_data, municipality_gdf, how='left', predicate='within')

    # Load external data
    pop_density_data = pd.read_excel('./data/external_data/PopDensity.xlsx')
    house_income_data = pd.read_excel('./data/external_data/HouseholdIncome.xlsx')

    # Convert columns to int type
    pop_density_data['Refnis'] = pop_density_data['Refnis'].astype(int)
    house_income_data['Refnis'] = house_income_data['CD_MUNTY_REFNIS'].astype(int)
    
    # Replace NaN values in 'cd_munty_refnis' column
    joined_data['cd_munty_refnis'] = joined_data['cd_munty_refnis'].fillna(-1)  # Use -1 as a placeholder value
    
    # Convert 'cd_munty_refnis' column to int type
    joined_data['cd_munty_refnis'] = joined_data['cd_munty_refnis'].astype(int)

    # Perform second join with population density data
    joined_data = joined_data.merge(pop_density_data, left_on='cd_munty_refnis', right_on='Refnis', how='left')
    joined_data = joined_data.merge(house_income_data, left_on='cd_munty_refnis', right_on='Refnis', how='left')

    # Convert the result to a DataFrame
    joined_data = pd.DataFrame(joined_data)

    joined_data.to_csv('./data/join_data.csv', index=False)

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
    cleaned_data = cleaned_data.dropna(subset=['Price', 'LivingArea', 'BedroomCount', 'Longitude', 'Latitude'])
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
    'ID', 'Street', 'HouseNumber', 'Box', 'City', 'Region', 'District', 'PropertyType', 
    'SaleType', 'BidStylePricing', 'KitchenType', 'Terrace', 'Garden', 'EPCScore', 
    'Condition', 'ListingExpirationDate', 'ListingCloseDate', 
    'PropertyUrl', 'ListingCreateDate', 'Property url', 'geometry', 'index_right', 'ogc_fid', 
    'ogc_fid0', 'cd_sector', 'tx_sector_descr_nl', 'tx_sector_descr_fr', 'tx_sector_descr_de', 
    'cd_sub_munty', 'tx_sub_munty_nl', 'tx_sub_munty_fr', 'tx_munty_dstr', 'cd_munty_refnis', 
    'tx_munty_descr_nl', 'tx_munty_descr_fr', 'tx_munty_descr_de', 'cd_dstr_refnis', 
    'tx_adm_dstr_descr_nl', 'tx_adm_dstr_descr_fr', 'tx_adm_dstr_descr_de', 'cd_prov_refnis', 
    'tx_prov_descr_nl', 'tx_prov_descr_fr', 'tx_prov_descr_de', 'cd_rgn_refnis', 
    'tx_rgn_descr_nl', 'tx_rgn_descr_fr', 'tx_rgn_descr_de', 'cd_country', 'cd_nuts_lvl1', 
    'cd_nuts_lvl2', 'cd_nuts_lvl3', 'ms_area_ha', 'ms_perimeter_m', 'dt_situation', 
    'Municipality FR', 'Municipality NL', 'Size in km²', 'Refnis_x', 
    'CD_MUNTY_REFNIS', 'MS_NBR_NON_ZERO_INC', 'MS_NBR_ZERO_INC', 
    'MS_TOT_NET_TAXABLE_INC', 'MS_NBR_TOT_NET_INC', 'MS_REAL_ESTATE_NET_INC', 
    'MS_NBR_REAL_ESTATE_NET_INC', 'MS_TOT_NET_MOV_ASS_INC', 'MS_NBR_NET_MOV_ASS_INC', 
    'MS_TOT_NET_VARIOUS_INC', 'MS_NBR_NET_VARIOUS_INC', 'MS_TOT_NET_PROF_INC', 'MS_NBR_NET_PROF_INC', 
    'MS_SEP_TAXABLE_INC', 'MS_NBR_SEP_TAXABLE_INC', 'MS_JOINT_TAXABLE_INC', 'MS_NBR_JOINT_TAXABLE_INC', 
    'MS_TOT_DEDUCT_SPEND', 'MS_NBR_DEDUCT_SPEND', 'MS_TOT_STATE_TAXES', 'MS_NBR_STATE_TAXES', 
    'MS_TOT_MUNICIP_TAXES', 'MS_NBR_MUNICIP_TAXES', 'MS_TOT_SUBURBS_TAXES', 'MS_NBR_SUBURBS_TAXES',
    'MS_TOT_TAXES', 'MS_NBR_TOT_TAXES', 'MS_TOT_RESIDENTS', 'Refnis_y', 'bookmarkCount'
    ]

    cleaned_data.drop(columns=columns_to_drop, inplace=True)

    # Save the cleaned data to a CSV file
    cleaned_data.to_csv('./data/clean_data.csv', index=False, encoding='utf-8')

    # Return the cleaned DataFrame
    return cleaned_data

def engineer_features(cleaned_data):
    engineered_data = cleaned_data.copy()

    # Task 14: Feature Engineering and Outlier Removal
    engineered_data['PricePerSqm'] = engineered_data['Price'] / engineered_data['LivingArea']

    # Create a new column called SqmPerBedroom
    engineered_data['SqmPerBedroom'] = engineered_data['LivingArea'] / engineered_data['BedroomCount']

    # Create a new column called AvgHouseholdIncome
    engineered_data['AvgHouseholdIncome'] = engineered_data['MS_TOT_NET_INC'] / engineered_data['Population']

    # Calculate z-scores within each group defined by 'PostalCode' and 'PropertySubType'
    z_scores = engineered_data.groupby(['PostalCode', 'PropertySubType'])[['PricePerSqm', 'SqmPerBedroom']].transform(stats.zscore)

    # Filter out rows where the absolute z-score is less than 3 for both columns
    engineered_data = engineered_data[(abs(z_scores['PricePerSqm']) < 3) & (abs(z_scores['SqmPerBedroom']) < 3)]

    # Drop columns if they exist
    columns_to_drop = ['PricePerSqm', 'PostalCode', 'Population', 'MS_TOT_NET_INC']
    engineered_data = engineered_data.drop(columns=[col for col in columns_to_drop if col in engineered_data.columns], errors='ignore')

    # Save the cleaned data to a CSV file
    engineered_data.to_csv('./data/engineered_data.csv', index=False, encoding='utf-8')

    # Return the cleaned DataFrame
    return engineered_data

def split_data(data, test_size=0.2, random_state=None):
    """
    Split the data into training and testing datasets.

    Parameters:
    data (DataFrame): The DataFrame containing the features and target variable
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2)
    random_state (int or None): Controls the shuffling applied to the data before splitting (default is None)

    Returns:
    tuple: A tuple containing train and test datasets (X_train, X_test, y_train, y_test)
    """
    # Extract features (X) and target variable (y)
    X = data.drop('Price', axis=1)
    y = data['Price']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def encode_data(cleaned_data, train_columns=None):
    """
    Encode categorical data using one-hot encoding.

    Parameters:
    cleaned_data (DataFrame): The cleaned DataFrame
    train_columns (list): List of column names in the training dataset (optional)

    Returns:
    DataFrame: The DataFrame with encoded categorical features
    """
   
    # Encode categorical data using one-hot encoding
    encoded_data = pd.get_dummies(cleaned_data, columns=['PropertySubType', 'Province'], dummy_na=False, drop_first=True)

    # Convert boolean columns to integer (0 or 1)
    bool_columns = encoded_data.select_dtypes(include=bool).columns
    encoded_data[bool_columns] = encoded_data[bool_columns].astype(int)

    # Ensure that the columns in the test dataset match the columns in the training dataset
    if train_columns is not None:
        # Add missing columns in test dataset, if any
        missing_columns = set(train_columns) - set(encoded_data.columns)
        for col in missing_columns:
            encoded_data[col] = 0

        # Remove additional columns in test dataset, if any
        extra_columns = set(encoded_data.columns) - set(train_columns)
        encoded_data = encoded_data[train_columns]

    # Save the encoded data to a CSV file
    encoded_data.to_csv('./data/encoded_data.csv', index=False, encoding='utf-8')

    return encoded_data

def impute_data(encoded_data):
    """
    Impute missing values in the DataFrame using KNN imputation.

    Parameters:
    encoded_data (DataFrame): The DataFrame with encoded categorical features

    Returns:
    DataFrame: The DataFrame with imputed missing values
    """
    # Create a copy of the encoded data
    imputed_data = encoded_data.copy()

    # Identify columns with missing values
    columns_with_missing_values = imputed_data.columns[imputed_data.isnull().any()].tolist()

    # Use KNN imputer to impute missing values
    imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed
    imputed_data[columns_with_missing_values] = imputer.fit_transform(imputed_data[columns_with_missing_values])

    # Save the imputed data to a CSV file
    imputed_data.to_csv('./data/imputed_data.csv', index=False, encoding='utf-8')

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

    # Save the standardized data to a CSV file
    standardized_data.to_csv('./data/standardized_data.csv', index=False, encoding='utf-8')

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
    elif model_type == "logarithmic_regression":
        from source.models import execute_logarithmic_regression
        metrics, y_test, y_pred = execute_logarithmic_regression(refresh_data)
    elif model_type == "auto_ml":
        from source.models import execute_auto_ml
        metrics, y_test, y_pred = execute_auto_ml(refresh_data)
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

def visualize_metrics(metrics, y_test, y_pred):
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

    # Format Mean Squared Error with commas for easier readability
    formatted_mse = "{:,.2f}".format(mse)

    # Print the metrics
    print("Evaluation Metrics:")
    print("Mean Squared Error:", formatted_mse)
    print("R-squared value:", f"{r_squared_percent:.2f}%")

    # Plot the metrics
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

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

    plt.tight_layout()
    plt.show()