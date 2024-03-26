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
    Province: str
    PropertySubType: str
    BedroomCount: int
    LivingArea: float
    KitchenType: Optional[str] 
    Furnished: Optional[int]
    Fireplace: Optional[int]
    TerraceArea: Optional[float] 
    GardenArea: Optional[float] 
    Facades: Optional[int] 
    SwimmingPool: Optional[int]  
    EnergyConsumptionPerSqm: Optional[float]
    Condition: Optional[str]
    Latitude: Optional[float] 
    Longitude: Optional[float] 

# Load the trained random forest model
model = joblib.load("./models/random_forest.pkl")

# Load preprocessing steps from JSON file
def load_preprocessing_steps(file_path):
    with open(file_path, 'r') as json_file:
        preprocessing_steps = json.load(json_file)
    return preprocessing_steps

# Apply preprocessing steps to input data
def apply_preprocessing(input_data, preprocessing_steps):
    processed_data = input_data.copy()
    for step, function in preprocessing_steps.items():
        module_name, function_name = function.rsplit('.', 1)
        module = importlib.import_module(module_name)
        process_function = getattr(module, function_name)
        processed_data = process_function(processed_data)
    return processed_data

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
    preprocessing_steps = load_preprocessing_steps("./preprocessing_steps.json")

    # Apply preprocessing steps
    preprocessed_data = apply_preprocessing(input_df, preprocessing_steps)

    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_data)
    corrected_predictions = np.power(10, predictions) - 1

    # Format predictions as currency
    formatted_predictions = [f'€{price:,.2f}' for price in corrected_predictions]

    return formatted_predictions
