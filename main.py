import time
from source.models import trainLinearRegression, trainRandomForest, trainLGBM

def main():
    """
    Main function to execute the Model Execution App.
    """
    print("Welcome to the Model Execution App!")

    # Ask which model to run
    print("Which model would you like to run?")
    print("1. Linear Regression")
    print("2. Random Forest")
    print("3. LightGBM")
    model_choice = int(input("Enter your choice (1/2/3): "))  # Prompting user for model choice

    # Ask if the data should be refreshed
    print("\nWould you like to refresh the data?")
    print("1. Yes")
    print("2. No")
    refresh_choice = int(input("Enter your choice (1/2): "))  # Prompting user for refresh choice

    refresh_data = True if refresh_choice == 1 else False  # Setting refresh_data based on user input

    # Execute the model
    start_time = time.time()  # Record start time of program execution
    print("\nProgram initiated:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))  # Print program start time
    
    if model_choice == 1:
        print("Model selected: Linear Regression")  # Print selected model
        trainLinearRegression(refresh_data)
    elif model_choice == 2:
        print("Model selected: Random Forest")  # Print selected model
        trainRandomForest(refresh_data)
    elif model_choice == 3:
        print("Model selected: LightGBM")  # Print selected model
        trainLGBM(refresh_data)
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

    # Print total runtime
    end_time = time.time()  # Record end time of program execution
    runtime = end_time - start_time  # Calculate total runtime
    print(f"Total Run Time: {runtime:.2f} seconds")  # Print total runtime

if __name__ == "__main__":
    main()