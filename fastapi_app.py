# reduce the input data scheme to features only
# modularize your preprocessing steps and wrap them with if the column exists statements
# test with 'uvicorn fastapi_app:app --reload' and swagger ui 'http://localhost:8000/redoc.'


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
    Street: Optional[str]
    HouseNumber: Optional[str] 
    Box: Optional[str] 
    Floor: Optional[float] 
    City: Optional[str]
    Region: Optional[str] 
    District: Optional[str] 
    Province: Optional[List[str]] 
    PropertyType: Optional[str] 
    PropertySubType: List[str]
    Price: Optional[float] 
    SaleType: Optional[str]
    BidStylePricing: Optional[float]
    ConstructionYear: Optional[float]
    BedroomCount: int
    LivingArea: float
    KitchenType: Optional[str] 
    Furnished: Optional[bool]
    Fireplace: Optional[bool]
    Terrace: Optional[float] 
    TerraceArea: Optional[float] 
    Garden: Optional[float] 
    GardenArea: Optional[float] 
    Facades: Optional[int] 
    SwimmingPool: Optional[bool]  
    Condition: Optional[List[str]] 
    EPCScore: Optional[str]  
    EnergyConsumptionPerSqm: Optional[float]
    Latitude: Optional[float] 
    Longitude: Optional[float] 
    ListingCreateDate: Optional[str] 
    ListingExpirationDate: Optional[str] 
    ListingCloseDate: Optional[str]  
    bookmarkCount: Optional[float] 
    ViewCount: Optional[float] 
    PropertyUrl: Optional[float]  
    Property_url: Optional[str]  

# Initialize FastAPI app
app = FastAPI()

# Load data to extract options
raw_data = pd.read_csv('./files/data/raw_data.csv')
province_options = raw_data['Province'].unique().tolist()
property_subtype_options = raw_data['PropertySubType'].unique().tolist()
condition_options = raw_data['Condition'].unique().tolist()
kitchen_type_options = raw_data['KitchenType'].unique().tolist()

# Define API endpoint for predicting property prices
@app.post("/predict/")
def predict_property_price(data: List[InputData]):
    input_df = pd.DataFrame([item.dict() for item in data])

    # Ensure the DataFrame has the same columns as the training data
    expected_columns = [field for field, field_info in InputData.__fields__.items() if field_info.is_required]
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

    # Perform preprocessing steps
    raw_data = data
    joined_data = join_data(raw_data)
    cleaned_data = clean_data(joined_data)
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
