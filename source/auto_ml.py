from source.preprocessing import clean_data, normalize_data
from source.utils import import_data, save_model, print_details
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
import time

def execute_auto_ml(refresh=False):
    """
    Execute the auto_ml model.

    Parameters:
    refresh (bool): Whether to refresh the data.

    Returns:
    None
    """
    start_time = time.time()
    print("Model selected: Auto ML")
    print("Program initiated:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    # Import and preprocess data
    raw_data = import_data(refresh)
    cleaned_data = clean_data(raw_data)
    normalized_data = normalize_data(cleaned_data)

    # Split features (X) and target variable (y)
    X = normalized_data.drop('Price', axis=1)
    y = normalized_data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the TPOT regressor
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)

    # Fit the TPOT regressor to the training data
    tpot.fit(X_train, y_train)

    # Evaluate the TPOT regressor's performance
    mse = tpot.score(X_test, y_test)
    print(f'Mean Squared Error: {mse}')

    # Save the TPOT regressor
    save_model(tpot.fitted_pipeline_, "auto_ml_model")

    print_details(start_time)