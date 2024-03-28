import os
import sys
import pandas as pd
import numpy as np
import joblib

# Function to apply preprocessing steps to input data
def apply_preprocessing(input_data, preprocessing_pipeline):
    processed_data = input_data.copy()
    processed_data = preprocessing_pipeline.transform(processed_data)
    return processed_data

# Function to predict the price using the trained LightGBM model
def predict_price(input_data):
    # Load the trained LightGBM model and preprocessing pipeline
    model_data = joblib.load("./models/light_gbm.pkl")
    model = model_data["model"]
    preprocessing_pipeline = model_data["preprocessing_pipeline"]

    # Apply preprocessing pipeline to input data
    preprocessed_data = apply_preprocessing(input_data, preprocessing_pipeline)

    # Make predictions
    predicted_price = model.predict(preprocessed_data)

    predicted_price = np.power(10, predicted_price) - 1

    return predicted_price[0]

if __name__ == "__main__":
    # Example input data for a new house
    new_house_data = {
        'PostalCode': 9940,
	    'Region': 'FLANDERS',
	    'District': 'Gent',
        'Province': 'East Flanders',
	    'PropertyType': 'House',
        'PropertySubType': 'House',
        'BedroomCount': 3,
        'LivingArea': 155,
        'KitchenType': 'Installed',
        'Furnished': 0,
        'Fireplace': 0,
	    'Terrace': 0,
        'TerraceArea': 0,
        'Garden': 1,
        'GardenArea': 35,
        'Facades': 3,
        'SwimmingPool': 0,
        'EnergyConsumptionPerSqm': 100,
        'Condition': 'Good',
	    'EPCScore': 'B',
        'Latitude': 51.1114671,
        'Longitude': 3.6997650
    }

    # Convert the dictionary to a DataFrame
    new_house_data = pd.DataFrame([new_house_data])

    # Predict the price of the new house
    predicted_price = predict_price(new_house_data)
    print("Predicted price of the new house:", predicted_price)