from source.utils import execute_model, print_details
import time

def main():
    print("Welcome to the Model Execution App!")

    # Ask which model to use
    model = input("Select a model to run (linear_regression, auto_ml, multi_linear_regression): ")

    # Ask if the data should be refreshed
    refresh_data = input("Would you like to refresh the data? (yes/no): ").lower() == 'yes'

    # Execute the selected model
    execute_model(model, refresh_data)

if __name__ == "__main__":
    main()