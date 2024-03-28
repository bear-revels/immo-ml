from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import json
import importlib
import numpy as np
from typing import List, Optional

# Define input data schema
class InputData(BaseModel):
    PostalCode: int
    Region: str
    District: str
    Province: str
    PropertyType: str
    PropertySubType: str
    BedroomCount: int
    LivingArea: float
    KitchenType: Optional[str] 
    Furnished: Optional[int]
    Fireplace: Optional[int]
    Terrace: int
    TerraceArea: Optional[float] 
    Garden: int
    GardenArea: Optional[float] 
    Facades: Optional[int] 
    SwimmingPool: Optional[int]  
    EnergyConsumptionPerSqm: Optional[float]
    Condition: Optional[str]
    EPCScore: Optional[str]
    Latitude: Optional[float] 
    Longitude: Optional[float] 

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

# Define FastAPI app
app = FastAPI()

# Define API endpoint for predicting property prices
@app.post("/predict/")
def predict_property_price(data: List[InputData]):
    input_df = pd.DataFrame([item.dict() for item in data])

    # Ensure the DataFrame has the same columns as the training data
    expected_columns = [field for field, field_info in InputData.__fields__.items() if field_info.is_required]
    missing_columns = set(expected_columns) - set(input_df.columns)
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")

    # Load preprocessing steps
    predicted_price = predict_price(input_df)
    corrected_predictions = np.power(10, predicted_price) - 1

    # Format predictions as currency
    formatted_predictions = [f'€{price:,.2f}' for price in corrected_predictions]

    return formatted_predictions