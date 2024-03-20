from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from source.utils import (import_data, clean_data, join_data, transform_features, engineer_features,
                          encode_data, impute_data, standardize_data, save_model, visualize_metrics)

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
    # Preprocess the dataset
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)
    cleaned_data = clean_data(joined_data)
    transformed_data = transform_features(cleaned_data)
    engineered_data = engineer_features(transformed_data)
    encoded_data = encode_data(engineered_data)

    # Split the data into features (X) and target (y)
    X = encoded_data.drop('Price', axis=1)
    y = encoded_data['Price']

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    standardized_train_data = standardize_data(X_train)
    standardized_test_data = standardize_data(X_test)
    
    # Impute missing values
    imputed_train_data = impute_data(standardized_train_data)
    imputed_test_data = impute_data(standardized_test_data)

    # Initialize Linear Regression model
    model = LinearRegression()
    
    # Fit the model to the training data
    model.fit(imputed_train_data, y_train)
    
    # Evaluate the model's performance
    y_pred = model.predict(imputed_test_data)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Save the model
    save_model(model, "linear_regression")

    # Visualize the metrics
    visualize_metrics({"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred)

    return {"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred

def execute_gradient_boosted_decision_tree(refresh_data):
    """
    Execute a Gradient Boosted Decision Tree model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    dict: Evaluation metrics of the model
    array-like: Actual target values
    array-like: Predicted target values
    """
    # Preprocess the dataset
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)
    cleaned_data = clean_data(joined_data)
    transformed_data = transform_features(cleaned_data)

    # Perform label encoding on categorical columns
    label_encoders = {}
    for column in transformed_data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        transformed_data[column] = label_encoders[column].fit_transform(transformed_data[column])

    # Split the data into features (X) and target (y)
    X = transformed_data.drop('Price', axis=1)
    y = transformed_data['Price']

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

    # Impute missing values in the training data with median values
    # Impute missing values
    imputed_train_data = impute_data(X_train)
    imputed_test_data = impute_data(X_test)

    # Initialize the Gradient Boosted Decision Tree model
    model = GradientBoostingRegressor(random_state=60)

    # Fit the model
    model.fit(imputed_train_data, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(imputed_test_data)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Save the model
    save_model(model, "gradient_boosted_decision_tree")

    # Visualize the metrics
    visualize_metrics({"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred)

    return {"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred

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
    # Preprocess the dataset
    raw_data = import_data(refresh_data)
    joined_data = join_data(raw_data)
    cleaned_data = clean_data(joined_data)
    transformed_data = transform_features(cleaned_data)

    # Perform label encoding on categorical columns
    label_encoders = {}
    for column in transformed_data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        transformed_data[column] = label_encoders[column].fit_transform(transformed_data[column])

    # Split the data into features (X) and target (y)
    X = transformed_data.drop('Price', axis=1)
    y = transformed_data['Price']

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

    # Impute missing values in the training data with median values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(random_state=60)

    # Fit the model
    model.fit(X_train_imputed, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test_imputed)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Save the model
    save_model(model, "random_forest_regressor")

    # Visualize the metrics
    visualize_metrics({"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred)
    
    return {"Mean Squared Error": mse, "R-squared value": r_squared}, y_test, y_pred