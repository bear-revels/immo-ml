from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import pandas as pd

def clean_data(raw_data):
    """
    Clean the raw data by performing several tasks:
    1. Drop rows with empty values in 'Price' and 'LivingArea' columns
    2. Remove duplicates in the 'ID' column and where all columns but 'ID' are equal
    3. Convert empty values to 0 for specified columns; assumption that if blank then 0
    4. Filter rows where SaleType == 'residential_sale' and BidStylePricing == 0
    5. Remove specified columns
    6. Adjust text format
    7. Remove leading and trailing spaces from string columns
    8. Replace the symbol '�' with 'e' in all string columns
    9. Fill missing values with None and convert specified columns to float64 type
    10. Convert specified columns to Int64 type
    11. Replace any ConstructionYear > current_year + 10 with None
    12. Trim text after and including '_' from the 'EPCScore' column
    13. Convert 'ListingCreateDate' to Date type with standard DD/MM/YYYY format
    14. Replace values less than or equal to 0 in 'EnergyConsumptionPerSqm' with 0
    15. Calculate 'TotalArea'
    16. Calculate 'PricePerLivingSquareMeter'
    17. Calculate 'PricePerTotalSquareMeter'
    18. Convert string values to numeric values using dictionaries for specified columns
    19. Impute null values for 'Floor' by using the median value of similar 'PropertySubType' and 'PostalCode'
    20. Impute null values for 'ConstructionYear' by using the median value of similar 'PropertySubType' and 'PostalCode'
    21. Impute null values for 'Facades' by using the median value of similar 'PropertySubType' and 'PostalCode'
    22. Impute null values for 'EnergyConsumptionPerSqm' by using the median value of similar 'EPCScore', 'PropertySubType', and 'PostalCode'
    23. Impute null values for 'Condition#' by using the median value of similar 'PropertySubType', 'PostalCode', and range of construction year within +/-10
    24. Impute null values for 'KitchenType#' by using the median value of similar 'PropertySubType' and 'PostalCode'

    Parameters:
    raw_data (DataFrame): The raw DataFrame to be cleaned
    """

    cleaned_data = raw_data.copy()

    # Task 1: Drop rows with empty values in 'Price' and 'LivingArea' columns
    cleaned_data.dropna(subset=['Price', 'LivingArea'], inplace=True)
    
    # Task 2: Remove duplicates in the 'ID' column and where all columns but 'ID' are equal
    cleaned_data.drop_duplicates(subset='ID', inplace=True)
    cleaned_data.drop_duplicates(subset=cleaned_data.columns.difference(['ID']), keep='first', inplace=True)

    # Task 3: Convert empty values to 0 for specified columns; assumption that if blank then 0
    columns_to_fill_with_zero = ['Furnished', 'Fireplace', 'TerraceArea', 'GardenArea', 'SwimmingPool', 'BidStylePricing', 'ViewCount', 'bookmarkCount']
    cleaned_data[columns_to_fill_with_zero] = cleaned_data[columns_to_fill_with_zero].fillna(0)

    # Task 4: Filter rows where SaleType == 'residential_sale' and BidStylePricing == 0
    cleaned_data = cleaned_data[(cleaned_data['SaleType'] == 'residential_sale') & (cleaned_data['BidStylePricing'] == 0)].copy()

    # Task 6: Adjust text format
    columns_to_str = ['PropertySubType', 'KitchenType', 'Condition', 'EPCScore']

    def adjust_text_format(x):
        if isinstance(x, str):
            return x.title()
        else:
            return x

    cleaned_data.loc[:, columns_to_str] = cleaned_data.loc[:, columns_to_str].map(adjust_text_format)

    # Task 7: Remove leading and trailing spaces from string columns
    cleaned_data.loc[:, columns_to_str] = cleaned_data.loc[:, columns_to_str].apply(lambda x: x.str.strip() if isinstance(x, str) else x)

    # Task 8: Replace the symbol '�' with 'e' in all string columns
    cleaned_data = cleaned_data.map(lambda x: x.replace('�', 'e') if isinstance(x, str) else x)

    # Task 9: Fill missing values with None and convert specified columns to float64 type
    columns_to_fill_with_none = ['EnergyConsumptionPerSqm']
    cleaned_data[columns_to_fill_with_none] = cleaned_data[columns_to_fill_with_none].where(cleaned_data[columns_to_fill_with_none].notna(), None)

    columns_to_float64 = ['TerraceArea', 'GardenArea', 'EnergyConsumptionPerSqm']
    cleaned_data[columns_to_float64] = cleaned_data[columns_to_float64].astype(float)

    # Task 10: Convert specified columns to Int64 type
    columns_to_int64 = ['PostalCode', 'ConstructionYear', 'Floor', 'Furnished', 'Fireplace','Facades', 'SwimmingPool', 'bookmarkCount', 'ViewCount']
    cleaned_data[columns_to_int64] = cleaned_data[columns_to_int64].astype(float).round().astype('Int64')

    # Task 11: Replace any ConstructionYear > current_year + 10 with None
    current_year = datetime.now().year
    max_construction_year = current_year + 10
    cleaned_data['ConstructionYear'] = cleaned_data['ConstructionYear'].where(cleaned_data['ConstructionYear'] <= max_construction_year, None)

    # Task 12: Trim text after and including '_' from the 'EPCScore' column
    cleaned_data['EPCScore'] = cleaned_data['EPCScore'].str.split('_').str[0]

    # Task 13: Convert 'ListingCreateDate' to integer timestamp
    cleaned_data['ListingCreateDate'] = cleaned_data['ListingCreateDate'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()))

    # Task 14: Replace values less than or equal to 0 in 'EnergyConsumptionPerSqm' with 0
    cleaned_data.loc[cleaned_data['EnergyConsumptionPerSqm'] < 0, 'EnergyConsumptionPerSqm'] = 0

    # Task 16: Calculate 'Price_sqm'
    cleaned_data['Price_sqm'] = (cleaned_data['Price'] / cleaned_data['LivingArea']).round().astype(int)

    # Task 18: Convert string values to numeric values using dictionaries for specified columns
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

    # Task 19: Impute null values for 'Floor'
    floor_imputer = SimpleImputer(strategy='median')
    cleaned_data['Floor'] = floor_imputer.fit_transform(cleaned_data[['Floor']])

    # Task 20: Impute null values for 'ConstructionYear'
    construction_year_imputer = SimpleImputer(strategy='median')
    cleaned_data['ConstructionYear'] = construction_year_imputer.fit_transform(cleaned_data[['ConstructionYear']])

    # Task 21: Impute null values for 'Facades'
    facades_imputer = SimpleImputer(strategy='median')
    cleaned_data['Facades'] = facades_imputer.fit_transform(cleaned_data[['Facades']])

    # Task 22: Impute null values for 'EnergyConsumptionPerSqm'
    energy_imputer = SimpleImputer(strategy='median')
    cleaned_data['EnergyConsumptionPerSqm'] = energy_imputer.fit_transform(cleaned_data[['EnergyConsumptionPerSqm']])

    # Task 23: Impute null values for 'Condition#'
    condition_imputer = SimpleImputer(strategy='median')
    cleaned_data['Condition#'] = condition_imputer.fit_transform(cleaned_data[['Condition#']])

    # Task 24: Impute null values for 'KitchenType#'
    kitchen_imputer = SimpleImputer(strategy='median')
    cleaned_data['KitchenType#'] = kitchen_imputer.fit_transform(cleaned_data[['KitchenType#']])

    # Task 25: Round float columns to the nearest whole number and convert to int64 type
    float_columns = cleaned_data.select_dtypes(include=['float64']).columns
    cleaned_data[float_columns] = cleaned_data[float_columns].round().astype('Int64')

    # Task 5: Remove specified columns
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

    # Task 25: Convert 'PropertySubType' categorical data into numeric features using one-hot encoding
    cleaned_data = pd.get_dummies(cleaned_data, columns=['PropertySubType'], prefix='PropertyType', dummy_na=False)
    cleaned_data = cleaned_data.astype(int)  # Convert boolean values to integers (1s and 0s)

    # Normalize and standardize the data
    cleaned_data = normalize_data(cleaned_data)

    # Save the cleaned data to a CSV file
    cleaned_data.to_csv('./data/clean_data.csv', index=False, encoding='utf-8')

    # Return the cleaned DataFrame
    return cleaned_data

def normalize_data(cleaned_data):
    # Select the columns to normalize
    columns_to_normalize = ['Floor', 'PostalCode', 'Price', 'ConstructionYear', 'LivingArea', 'Furnished', 'Fireplace', 'TerraceArea', 'GardenArea', 'Facades', 'SwimmingPool', 'EnergyConsumptionPerSqm', 'ListingCreateDate', 'bookmarkCount', 'ViewCount', 'Price_sqm', 'Condition#', 'KitchenType#']

    # Initialize the StandardScaler and MinMaxScaler
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    # Fit and transform the data with StandardScaler
    cleaned_data[columns_to_normalize] = scaler_standard.fit_transform(cleaned_data[columns_to_normalize])

    # Fit and transform the data with MinMaxScaler
    cleaned_data[columns_to_normalize] = scaler_minmax.fit_transform(cleaned_data[columns_to_normalize])

    return cleaned_data