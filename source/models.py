from source.utils import import_data, clean_data, normalize_data, save_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

def execute_linear_regression(refresh_data):
    """
    Execute a linear regression model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    None
    """
    # Import and preprocess data
    raw_data = import_data(refresh_data)
    cleaned_data = clean_data(raw_data)

    # Split the data into train and test sets
    train_data, test_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)

    # Normalize train and test sets separately
    normalized_train_data = normalize_data(train_data)
    normalized_test_data = normalize_data(test_data)

    # Split features (X) and target variable (y) for train and test sets
    X_train = normalized_train_data.drop('Price', axis=1)
    y_train = normalized_train_data['Price']
    X_test = normalized_test_data.drop('Price', axis=1)
    y_test = normalized_test_data['Price']

    # Instantiate the linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Calculate the R-squared value
    r_squared = r2_score(y_test, y_pred)
    print(f'R-squared value: {r_squared:.2%}')

    # Save the model
    save_model(model, "linear_regression_model")

def execute_multi_linear_regression(refresh_data):
    """
    Execute a multi-linear regression model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    None
    """
    # Import and preprocess data
    raw_data = import_data(refresh_data)
    cleaned_data = clean_data(raw_data)

    # Split the data into train and test sets
    train_data, test_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)

    # Normalize train and test sets separately
    normalized_train_data = normalize_data(train_data)
    normalized_test_data = normalize_data(test_data)

    # Split features (X) and target variable (y) for train and test sets
    X_train = normalized_train_data.drop('Price', axis=1)
    y_train = normalized_train_data['Price']
    X_test = normalized_test_data.drop('Price', axis=1)
    y_test = normalized_test_data['Price']

    # Instantiate the multi-linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Calculate the R-squared value
    r_squared = r2_score(y_test, y_pred)
    print(f'R-squared value: {r_squared:.2%}')

    # Save the model
    save_model(model, "multi_linear_regression_model")

def execute_auto_ml(refresh_data):
    """
    Execute an auto-ML model.

    Parameters:
    refresh_data (bool): Whether to refresh the data.

    Returns:
    None
    """
    # Import and preprocess data
    raw_data = import_data(refresh_data)
    cleaned_data = clean_data(raw_data)

    # Split the data into train and test sets
    train_data, test_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)

    # Normalize train and test sets separately
    normalized_train_data = normalize_data(train_data)
    normalized_test_data = normalize_data(test_data)

    # Split features (X) and target variable (y) for train and test sets
    X_train = normalized_train_data.drop('Price', axis=1)
    y_train = normalized_train_data['Price']
    X_test = normalized_test_data.drop('Price', axis=1)
    y_test = normalized_test_data['Price']

    # Instantiate the TPOT regressor
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)

    # Fit the TPOT regressor to the training data
    tpot.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = tpot.predict(X_test)

    # Evaluate the TPOT regressor's performance
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the TPOT regressor
    save_model(tpot.fitted_pipeline_, "auto_ml_model")