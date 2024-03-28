from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from source.utils import (import_data, clean_data, join_data, transform_features, engineer_features,
                          encode_data)
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os

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
    engineered_data = engineer_features(transformed_data)
    encoded_data = encode_data(engineered_data)

    print(encoded_data.info())

    # Split the data into features (X) and target (y)
    X = encoded_data.drop('Price', axis=1)
    y = encoded_data['Price']

    # Split the data into training and testing datasets
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
    save_model(model, "random_forest")

    # Save preprocessing steps to a JSON file
    preprocessing_steps = {
        "join_data": "source.utils.join_data",
        "clean_data": "source.utils.clean_data",
        "transform_features": "source.utils.transform_features",
        "engineer_features": "source.utils.engineer_features",
        "encode_data": "source.utils.encode_data"
    }
    with open('./source/preprocessing_steps.json', 'w') as json_file:
        json.dump(preprocessing_steps, json_file)

    # Visualize the metrics
    visualize_metrics({"Mean Absolute Error": mae, "R-squared value": r_squared}, y_test, y_pred)
    
    return {"Mean Absolute Error": mae, "R-squared value": r_squared}, y_test, y_pred

def save_model(model, filename):
    """
    Save a machine learning model to a file.

    Parameters:
    model: The machine learning model to be saved
    filename (str): The name of the file to save the model to
    """
    if not os.path.exists("./models"):
        os.makedirs("./models")
    filepath = os.path.join("./models", filename + ".pkl")
    joblib.dump(model, filepath)
    print(f"Model saved as {filepath}")

def visualize_metrics(metrics, y_test, y_pred):
    """
    Print the evaluation metrics of a model and visualize the predicted vs actual values.

    Parameters:
    metrics (dict): Dictionary containing evaluation metrics
    y_test (array-like): True target values
    y_pred (array-like): Predicted target values
    """
    # Extract metric values
    mae = metrics.get("Mean Absolute Error")
    r_squared = metrics.get("R-squared value")

    # Convert R-squared value to percentage
    r_squared_percent = round(r_squared * 100, 2)

    # Format Mean Squared Error with commas and two decimal places
    formatted_mae = "{:,.2f}".format(mae)

    # Print the metrics
    print("Evaluation Metrics:")
    print("Mean Absolute Error:", formatted_mae)
    print("R-squared value:", f"{r_squared_percent:.2f}%")

    # Plot the metrics
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Mean Squared Error
    ax[0].bar(["Mean Absolute Error"], [mae], color='blue')
    ax[0].set_title(f"Mean Absolute Error: {formatted_mae}")

    # Plot R-squared value
    ax[1].bar(["R-squared value"], [r_squared_percent], color='green')
    ax[1].set_title(f"R-squared value: {r_squared_percent:.2f}%")
    ax[1].set_ylim([0, 100])  # Set y-axis limits to 0 and 100

    # Plot predicted vs actual values
    ax[2].scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Plot diagonal line
    ax[2].set_xlabel('Actual')
    ax[2].set_ylabel('Predicted')
    ax[2].set_title('Predicted vs Actual')

    plt.tight_layout()
    plt.show()