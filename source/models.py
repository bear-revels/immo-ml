from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tpot import TPOTRegressor
from source.utils import import_data, clean_data, join_data, engineer_features, split_data, encode_data, impute_data, standardize_data, save_model
import numpy as np

def execute_linear_regression(refresh_data):
    """
    Execute a linear regression model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    dict: Evaluation metrics of the model
    """
    # Import and join data
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)

    # Clean and engineer the joined data
    cleaned_data = clean_data(joined_data)
    engineered_data = engineer_features(cleaned_data)

    # Split the cleaned data into training and testing datasets
    X_train, X_test, y_train, y_test = split_data(engineered_data)
    
    # Encode categorical data
    encoded_train_data = encode_data(X_train)  # Encode training data
    encoded_test_data = encode_data(X_test, train_columns=encoded_train_data.columns.tolist())  # Encode test data
    
    # Standardize the data
    standardized_train_data = standardize_data(encoded_train_data)
    standardized_test_data = standardize_data(encoded_test_data)
    
    # Impute missing values
    imputed_train_data = impute_data(standardized_train_data)
    imputed_test_data = impute_data(standardized_test_data)

    # Initiate the linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(imputed_train_data, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(imputed_test_data)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Save the model
    save_model(model, "linear_regression_model")

    return {"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred

def execute_logarithmic_regression(refresh_data):
    """
    Execute a logarithmic regression model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    dict: Evaluation metrics of the model
    array-like: Actual target values
    array-like: Predicted target values
    """
    # Import and join data
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)

    # Clean and engineer the joined data
    cleaned_data = clean_data(joined_data)
    engineered_data = engineer_features(cleaned_data)

    # Split the cleaned data into training and testing datasets
    X_train, X_test, y_train, y_test = split_data(engineered_data)
    
    # Encode categorical data
    encoded_train_data = encode_data(X_train)  # Encode training data
    encoded_test_data = encode_data(X_test, train_columns=encoded_train_data.columns.tolist())  # Encode test data
    
    # Standardize the data
    standardized_train_data = standardize_data(encoded_train_data)
    standardized_test_data = standardize_data(encoded_test_data)
    
    # Impute missing values
    imputed_train_data = impute_data(standardized_train_data)
    imputed_test_data = impute_data(standardized_test_data)

    # Transform the target variable using natural logarithm
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    # Initiate the linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(imputed_train_data, y_train_log)

    # Make predictions
    y_pred_log = model.predict(imputed_test_data)

    # Transform the predicted values back to original scale
    y_pred = np.exp(y_pred_log)
    y_test_original = np.exp(y_test_log)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test_original, y_pred)
    r_squared = r2_score(y_test_original, y_pred)

    # Save the model
    save_model(model, "logarithmic_regression_model")

    # Return evaluation metrics and predictions
    return {"Mean Squared Error": mse, "R-squared value": r_squared}, y_test_original, y_pred

def execute_auto_ml(refresh_data):
    """
    Execute an auto-ML model using TPOT.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    dict: Evaluation metrics of the model
    """
    # Import and join data
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)

    # Clean and engineer the joined data
    cleaned_data = clean_data(joined_data)
    engineered_data = engineer_features(cleaned_data)

    # Split the cleaned data into training and testing datasets
    X_train, X_test, y_train, y_test = split_data(engineered_data)
    
    # Encode categorical data
    encoded_train_data = encode_data(X_train)  # Encode training data
    encoded_test_data = encode_data(X_test, train_columns=encoded_train_data.columns.tolist())  # Encode test data
    
    # Standardize the data
    standardized_train_data = standardize_data(encoded_train_data)
    standardized_test_data = standardize_data(encoded_test_data)
    
    # Impute missing values
    imputed_train_data = impute_data(standardized_train_data)
    imputed_test_data = impute_data(standardized_test_data)

    # Initiate the TPOT regressor
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)

    # Fit the TPOT regressor to the training data
    tpot.fit(imputed_train_data, y_train)

    # Make predictions on the testing data
    y_pred = tpot.predict(imputed_test_data)

    # Evaluate the TPOT regressor's performance
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Save the TPOT regressor
    save_model(tpot.fitted_pipeline_, "auto_ml_model")

    # Export optimized pipeline to a text file
    tpot.export('./source/optimized_pipeline.py')

    return {"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred