import time
import joblib
import os
import pandas as pd

def print_details(start_time):
    """
    Print program details including initiation time and runtime.

    Parameters:
    start_time (float): The timestamp when the program started.
    """
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total Run Time: {runtime:.2f} seconds")

def import_data(refresh=False):
    if refresh:
        print("Loading and preprocessing new data...")
        raw_data = pd.read_csv("https://raw.githubusercontent.com/bear-revels/immo-eliza-scraping-Python_Pricers/main/data/all_property_details.csv", dtype={'PostalCode': str})
        raw_data.to_csv('./data/raw_data.csv', index=False, encoding='utf-8')
    else:
        print("Preprocessing the existing data...")
        raw_data = pd.read_csv('./data/raw_data.csv')
    return raw_data

def load_model(filename):
    filepath = os.path.join("./models", filename + ".pkl")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def save_model(model, filename):
    if not os.path.exists("./models"):
        os.makedirs("./models")
    filepath = os.path.join("./models", filename + ".pkl")
    joblib.dump(model, filepath)
    print(f"Model saved as {filepath}")

def execute_model(selected_model, refresh=False):
    if selected_model == "linear_regression":
        from source.linear_regression import execute_linear_regression
        execute_linear_regression(refresh)
    elif selected_model == "auto_ml":
        from source.auto_ml import execute_auto_ml
        execute_auto_ml(refresh)
    elif selected_model == "multi_linear_regression":
        from source.multi_linear_regression import execute_multi_linear_regression
        execute_multi_linear_regression(refresh)
    else:
        print("Selected model not found.")