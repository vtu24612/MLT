import pandas as pd
import sqlite3
import os

# --- Configuration ---
CSV_FILE = 'steam.csv'
DATABASE_FILE = 'steam_database.db'
TABLE_NAME = 'games'

def create_database_from_csv():
    """
    Reads data from a CSV file and loads it into a new SQLite database table.
    If the database file already exists, it will be replaced.
    """
    # --- 1. Check if the CSV file exists ---
    if not os.path.exists(CSV_FILE):
        print(f"Error: The file '{CSV_FILE}' was not found in this directory.")
        print("Please make sure your CSV file is in the same folder as this script.")
        return

    try:
        # --- 2. Load the CSV data using pandas ---
        print(f"Reading data from '{CSV_FILE}'...")
        df = pd.read_csv(CSV_FILE)
        print("CSV data loaded successfully.")

        # --- 3. Create a connection to the SQLite database ---
        # The connect() function will create the database file if it doesn't exist.
        conn = sqlite3.connect(DATABASE_FILE)
        print(f"Connecting to database '{DATABASE_FILE}'...")

        # --- 4. Write the pandas DataFrame to a SQL table ---
        # if_exists='replace': This will drop the table first if it already exists.
        # This is useful for re-running the script to update the data.
        print(f"Writing data to table '{TABLE_NAME}'...")
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        print("Data has been successfully written to the database.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- 5. Close the database connection ---
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")

# This block ensures the function runs only when the script is executed directly
if __name__ == "__main__":
    create_database_from_csv()
