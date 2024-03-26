import time
from source.models import execute_random_forest

def main():
    """
    Main function to execute the Model Execution App.
    """
    print("Welcome to the Model Execution App!")

    # Ask if the data should be refreshed
    print("Would you like to refresh the data?")
    print("1. Yes")
    print("2. No")
    refresh_choice = int(input("Enter your choice (1/2): "))  # Prompting user for refresh choice

    refresh_data = True if refresh_choice == 1 else False  # Setting refresh_data based on user input

    # Execute the model
    start_time = time.time()  # Record start time of program execution
    print("Program initiated:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))  # Print program start time
    print("Model selected: random_forest")  # Print selected model

    # Execute the random forest model function
    execute_random_forest(refresh_data)  # Executing model function

    # Print total runtime
    end_time = time.time()  # Record end time of program execution
    runtime = end_time - start_time  # Calculate total runtime
    print(f"Total Run Time: {runtime:.2f} seconds")  # Print total runtime

if __name__ == "__main__":
    main()