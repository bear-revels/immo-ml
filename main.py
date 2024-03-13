from source.dataprep import execute_data_prep

def main():
    refresh_input = input("Would you like to refresh the data? (yes/no): ").lower()
    refresh_data = refresh_input == "yes"
    raw_data, cleaned_data = execute_data_prep(refresh=refresh_data)

if __name__ == "__main__":
    main()