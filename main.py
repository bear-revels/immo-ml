import time
from source.utils import execute_model

def main():
    """
    Main function to execute the Model Execution App.
    """
    print("Welcome to the Model Execution App!")

    # Ask which model to use
    print("Select a model to run:")
    print("1. Linear Regression")
    print("2. Gradient Boosted Decision Tree")
    print("3. Random Forest")
    choice = int(input("Enter your choice (1/2/3): "))  # Prompting user for model choice

    models = {  # Dictionary mapping user input to model names
        1: "linear_regression",
        2: "gradient_boosted_decision_tree",
        3: "random_forest"
    }

    if choice not in models:  # Checking if the choice is valid
        print("Invalid choice. Please select a valid option.")
        return

    model = models[choice]  # Retrieving the selected model name based on the selection

    # Ask if the data should be refreshed
    print("Would you like to refresh the data?")
    print("1. Yes")
    print("2. No")
    refresh_choice = int(input("Enter your choice (1/2): "))  # Prompting user for refresh choice

    refresh_data = True if refresh_choice == 1 else False  # Setting refresh_data based on user input

    # Ask for comments
    comments = input("Would you like to add any comments? (Press Enter to skip): ")  # Prompting user for comments

    # Execute the model
    start_time = time.time()  # Record start time of program execution
    print("Program initiated:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))  # Print program start time
    print(f"Model selected: {model}")  # Print selected model

    # Execute the selected model function
    metrics, y_test, y_pred = execute_model(model, refresh_data)  # Executing model function

    # Print total runtime
    end_time = time.time()  # Record end time of program execution
    runtime = end_time - start_time  # Calculate total runtime
    print(f"Total Run Time: {runtime:.2f} seconds")  # Print total runtime

if __name__ == "__main__":
    main()