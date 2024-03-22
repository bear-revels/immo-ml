from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from source.utils import (clean_data, import_data, join_data, transform_features)
from typing import List, Optional

# Load the trained random forest model
model = joblib.load("models/random_forest.pkl")

# Define input data schema
class InputData(BaseModel):
    PostalCode: int
    Province: List[str]
    PropertySubType: List[str]
    LivingArea: float
    BedroomCount: int
    Furnished: Optional[bool]
    Fireplace: Optional[bool]
    TerraceArea: Optional[float]
    GardenArea: Optional[float]
    Facades: Optional[int]
    SwimmingPool: Optional[bool]
    EnergyConsumptionPerSqm: Optional[float]
    PopDensity: Optional[float]
    MedianPropertyValue: Optional[float]
    NetIncomePerResident: Optional[float]
    Condition: Optional[int]
    KitchenType: Optional[int]

# Initialize FastAPI app
app = FastAPI()

# Load data to extract options
raw_data = import_data(refresh_data=False)
province_options = raw_data['Province'].unique().tolist()
property_subtype_options = raw_data['PropertySubType'].unique().tolist()
condition_options = raw_data['Condition'].unique().tolist()
kitchen_type_options = raw_data['KitchenType'].unique().tolist()

# Define API endpoint for predicting property prices
@app.post("/predict/")
def predict_property_price(data: List[InputData]):
    input_df = pd.DataFrame([item.dict() for item in data])

    # Ensure the DataFrame has the same columns as the training data
    expected_columns = [field for field, field_info in InputData.__fields__.items() if field_info.required]
    missing_columns = set(expected_columns) - set(input_df.columns)
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")

    # Preprocess the input data
    preprocessed_data = preprocess_data(input_df)

    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_data)

    # Convert predictions to list
    predicted_prices = predictions.tolist()

    return predicted_prices

def preprocess_data(data: pd.DataFrame):
    # Validate input data against extracted options
    for attr in InputData.__fields__:
        if attr not in data.columns:
            raise HTTPException(status_code=400, detail=f"Invalid attribute: {attr}")

    for attr, options in [('Province', province_options), ('PropertySubType', property_subtype_options), 
                        ('Condition', condition_options), ('KitchenType', kitchen_type_options)]:
        if attr not in data.columns or not data[attr].isin(options).all():
            raise HTTPException(status_code=400, detail=f"Invalid {attr} values")

    # Perform preprocessing steps
    cleaned_data = clean_data(data)
    transformed_data = transform_features(cleaned_data)

    # Perform label encoding on categorical columns
    categorical_columns = transformed_data.select_dtypes(include=['object']).columns
    label_encoders = {column: LabelEncoder().fit(transformed_data[column]) for column in categorical_columns}
    for column, encoder in label_encoders.items():
        transformed_data[column] = encoder.transform(transformed_data[column])

    # Impute missing values in the training data with median values
    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(transformed_data)

    # Return preprocessed data
    return pd.DataFrame(imputed_data, columns=transformed_data.columns)
