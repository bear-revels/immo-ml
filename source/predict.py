import pandas as pd
import joblib
import json
import importlib
import numpy as np

# Function to load preprocessing steps from JSON file
def load_preprocessing_steps(file_path):
    with open(file_path, 'r') as json_file:
        preprocessing_steps = json.load(json_file)
    return preprocessing_steps

# Function to apply preprocessing steps to input data
def apply_preprocessing(input_data, preprocessing_steps):
    processed_data = input_data.copy()
    for step, function in preprocessing_steps.items():
        module_name, function_name = function.rsplit('.', 1)
        module = importlib.import_module(module_name)
        process_function = getattr(module, function_name)
        processed_data = process_function(processed_data)
    return processed_data

# Function to predict the price using the trained Random Forest model
def predict_price(input_data):
    # Load the trained Random Forest model
    model = joblib.load("./models/random_forest.pkl")

    # Load preprocessing steps from JSON file
    preprocessing_steps = load_preprocessing_steps("./preprocessing_steps.json")

    # Apply preprocessing steps to input data
    preprocessed_data = apply_preprocessing(input_data, preprocessing_steps)

    print(preprocessed_data.info())

    # Make predictions
    predicted_price = model.predict(preprocessed_data)

    predicted_price = np.power(10, predicted_price) - 1

    return predicted_price[0]

if __name__ == "__main__":
    # Example input data for a new house
    new_house_data = {
    'PostalCode': 9940,
    'Province': 'East Flanders',
    'PropertySubType': 'House',
    'BedroomCount': 3,
    'LivingArea': 155,
    'KitchenType': 'Installed',
    'Furnished': 0,
    'Fireplace': 0,
    'TerraceArea': 0,
    'GardenArea': 35,
    'Facades': 3,
    'SwimmingPool': 0,
    'EnergyConsumptionPerSqm': 100,
    'Condition': 'Good',
    'Latitude': 511114671,  # Converted to float
    'Longitude': 36997650   # Converted to float
}

    # Convert the dictionary to a DataFrame
    new_house_data = pd.DataFrame([new_house_data])

    # Specify data types for certain columns
    data_types = {
        'PostalCode': 'int',
        'BedroomCount': 'int',
        'LivingArea': 'int',
        'Furnished': 'int',
        'Fireplace': 'int',
        'TerraceArea': 'int',
        'GardenArea': 'int',
        'Facades': 'int',
        'SwimmingPool': 'int',
        'EnergyConsumptionPerSqm': 'int',
        'Latitude': 'float',
        'Longitude': 'float'
    }

    # Convert columns to specified data types
    for col, dtype in data_types.items():
        new_house_data[col] = new_house_data[col].astype(dtype)

    # Predict the price of the new house
    predicted_price = predict_price(new_house_data)
    print("Predicted price of the new house:", predicted_price)