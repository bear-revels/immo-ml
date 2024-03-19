from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from source.utils import import_data, clean_data, join_data, transform_features, engineer_features, split_data, encode_data, impute_data, standardize_data, save_model
import numpy as np
import pandas as pd

def execute_linear_regression(refresh_data):
    """
    Execute a linear regression model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    dict: Evaluation metrics of the model
    array-like: Actual target values
    array-like: Predicted target values
    """
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)
    cleaned_data = clean_data(joined_data)
    transformed_data = transform_features(cleaned_data)
    engineered_data = engineer_features(transformed_data)

    # Encode categorical data
    encoded_data = encode_data(engineered_data)

    X_train, X_test, y_train, y_test = split_data(encoded_data)
    
    # Standardize the data
    standardized_train_data = standardize_data(X_train)
    standardized_test_data = standardize_data(X_test)
    
    # Impute missing values
    imputed_train_data = impute_data(standardized_train_data)
    imputed_test_data = impute_data(standardized_test_data)
    
    model = LinearRegression()
    
    # Fit the model to the training data
    model.fit(imputed_train_data, y_train)
    
    # Evaluate the model's performance
    y_pred = model.predict(imputed_test_data)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, standardized_train_data, y_train, cv=5, scoring='r2')
    mean_cv_score = np.mean(cv_scores)

    # Save the model
    save_model(model, "linear_regression_model")

    return {"Mean Squared Error": mse, "R-squared value": r_squared, "Mean CV R-squared value": mean_cv_score}, y_test, y_pred

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
    transformed_data = engineer_features(cleaned_data)

    # Split the cleaned data into training and testing datasets
    X_train, X_test, y_train, y_test = split_data(transformed_data)
    
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


def execute_random_forest(refresh_data):
    """
    Execute a Random Forest Regressor model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    dict: Evaluation metrics of the model
    array-like: Actual target values
    array-like: Predicted target values
    """
    # Import and join the data
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)
    
    # Clean and engineer the joined data
    cleaned_data = clean_data(joined_data)
    transformed_data = transform_features(cleaned_data)

    # Perform label encoding on categorical columns
    label_encoders = {}
    for column in transformed_data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        transformed_data[column] = label_encoders[column].fit_transform(transformed_data[column])

    # Split the cleaned and encoded data into features (X) and target (y)
    X = transformed_data.drop('Price', axis=1)  # Assuming 'target_column' is the name of your target column
    y = transformed_data['Price']

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(random_state=60)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    actual_mse = 10**mse
    r_squared = r2_score(y_test, y_pred)

    # Save the model
    save_model(model, "random_forest_regressor_model")
    
    # Concatenate the test dataset with actual and predicted values
    test_data_with_predictions = pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')], axis=1)

    # Write the DataFrame to a CSV file
    test_data_with_predictions.to_csv('./data/actual_vs_pred.csv', index=False)

    return {"Mean Squared Error": actual_mse, "R-squared value": r_squared}, y_test, y_pred