from source.preprocessing import clean_data, normalize_data
from source.utils import import_data, save_model, print_details
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import time

def execute_linear_regression(refresh=False):
    """
    Execute the linear regression model.
    
    Parameters:
    refresh (bool): Whether to refresh the data
    
    Returns:
    None
    """
    start_time = time.time()
    print("Model selected: Linear Regression")
    print("Program initiated:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    # Import and preprocess data
    raw_data = import_data(refresh)
    cleaned_data = clean_data(raw_data)
    normalized_data = normalize_data(cleaned_data)

    # Selecting only the 'LivingArea' column as predictor variable
    X = normalized_data[['LivingArea']]
    y = normalized_data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Print the R-squared value as a percentage with two decimal places
    print(f'R-squared value: {r_squared:.2%}')

    # Save the model
    save_model(model, "linear_regression_model")

    print_details(start_time)