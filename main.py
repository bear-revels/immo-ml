from source.utils import execute_model  # Importing the execute_model function from source.utils module
import time

def main():
    """
    Main function to execute the Model Execution App.
    """
    print("Welcome to the Model Execution App!")

    # Ask which model to use
    model = input("Select a model to run (linear_regression, auto_ml, multi_linear_regression): ")

    # Ask if the data should be refreshed
    refresh_data = input("Would you like to refresh the data? (yes/no): ").lower() == 'yes'

    start_time = time.time()
    print("Program initiated:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print(f"Model selected: {model}")

    # Execute the selected model function
    execute_model(model, refresh_data)

    # Print total runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total Run Time: {runtime:.2f} seconds")

if __name__ == "__main__":
    main()