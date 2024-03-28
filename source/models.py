import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from source.utils import import_data, preprocessing, save_model, visualize_metrics

def trainLinearRegression(refresh_data):
    # Load raw data
    raw_data = import_data(refresh_data)

    # Apply preprocessing pipeline
    preprocessed_data = preprocessing.fit_transform(raw_data)

    # Separate features and target variable
    X = preprocessed_data.drop(columns=['Price'])
    y = preprocessed_data['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values in X_train and X_test with median strategy
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Initialize Linear Regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train_imputed, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test_imputed)

    # Apply inverse log transformation to y_test and y_pred
    adj_y_pred = np.power(10, y_pred) - 1
    adj_y_test = np.power(10, y_test) - 1

    # Evaluate the model
    mae = mean_absolute_error(adj_y_test, adj_y_pred)
    r_squared = r2_score(adj_y_test, adj_y_pred)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r_squared)

    # Save the model
    save_model(model, "linear_regression", None, preprocessing)
    visualize_metrics({"Mean Absolute Error": mae, "R-squared value": r_squared}, adj_y_test, adj_y_pred)

def trainRandomForest(refresh_data):
    """
    Execute a Random Forest Regressor model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    dict: Evaluation metrics of the model
    array-like: Actual target values
    array-like: Predicted target values
    """
    # Preprocess the dataset using the full pipeline
    raw_data = import_data(refresh_data)  # Assuming this function exists
    preprocessed_data = preprocessing.fit_transform(raw_data)

    # Separate features and target variable
    X = preprocessed_data.drop(columns=['Price'])
    y = preprocessed_data['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(random_state=60)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Apply inverse log transformation to y_test and y_pred
    y_pred = np.power(10, y_pred) - 1
    y_test = np.power(10, y_test) - 1

    # Evaluate the model's performance
    mae = mean_absolute_error(y_pred, y_test)
    r_squared = r2_score(y_test, y_pred)

    # Save the model
    save_model(model, "random_forest", None, preprocessing)

    # Visualize the metrics
    visualize_metrics({"Mean Absolute Error": mae, "R-squared value": r_squared}, y_test, y_pred)
    
    return {"Mean Absolute Error": mae, "R-squared value": r_squared}, y_test, y_pred

def trainLGBM(refresh_data):
    # Load raw data
    raw_data = import_data(refresh_data)

    # Apply preprocessing pipeline
    preprocessed_data = preprocessing.fit_transform(raw_data)

    # Separate features and target variable
    X = preprocessed_data.drop(columns=['Price'])
    y = preprocessed_data['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters
    hyperparameters = {
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 1.0,
        'importance_type': 'split',
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 300,
        'n_jobs': None,
        'num_leaves': 100,
        'objective': None,
        'random_state': None,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'subsample': 1.0,
        'subsample_for_bin': 200000,
        'subsample_freq': 0,
        'force_col_wise': False,
        'force_row_wise': True,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }

    # Initialize LGBMRegressor with specified hyperparameters
    model = LGBMRegressor(**hyperparameters, verbose=-1)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Apply inverse log transformation to y_test and y_pred
    adj_y_pred = np.power(10, y_pred) - 1
    adj_y_test = np.power(10, y_test) - 1

    # Evaluate the model
    mae = mean_absolute_error(adj_y_test, adj_y_pred)
    r_squared = r2_score(adj_y_test, adj_y_pred)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r_squared)

    # Save the model
    save_model(model, "light_gbm", hyperparameters, preprocessing)
    visualize_metrics({"Mean Absolute Error": mae, "R-squared value": r_squared}, adj_y_test, adj_y_pred)