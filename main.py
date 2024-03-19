import time
from source.utils import execute_model, visualize_performance

def main():
    print("Welcome to the Model Execution App!")

    # Ask which model to use
    print("Select a model to run:")
    print("1. Linear Regression")
    print("2. Logarithmic Regression")
    print("3. Random Forest")
    choice = int(input("Enter your choice (1/2/3): "))

    models = {
        1: "linear_regression",
        2: "logarithmic_regression",
        3: "random_forest"
    }

    if choice not in models:
        print("Invalid choice. Please select a valid option.")
        return

    model = models[choice]

    # Ask if the data should be refreshed
    print("Would you like to refresh the data?")
    print("1. Yes")
    print("2. No")
    refresh_choice = int(input("Enter your choice (1/2): "))

    refresh_data = True if refresh_choice == 1 else False

    # Ask for comments
    comments = input("Would you like to add any comments? (Press Enter to skip): ")

    # Execute the model
    start_time = time.time()
    print("Program initiated:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print(f"Model selected: {model}")

    # Execute the selected model function
    metrics, y_test, y_pred = execute_model(model, refresh_data)

    # Print total runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total Run Time: {runtime:.2f} seconds")

    # Visualize model evaluation metrics
    if metrics:
        visualize_performance(metrics, y_test, y_pred, comments)

if __name__ == "__main__":
    main()