import pandas as pd
import sqlite3
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# --- 1. Database Setup ---
def setup_database(csv_path='steam.csv', db_path='steam_games.db'):
    """
    Reads the steam.csv file, cleans the data, and loads it into a SQLite database.
    """
    if os.path.exists(db_path):
        print(f"Database '{db_path}' already exists. Skipping creation.")
        return

    try:
        df = pd.read_csv(csv_path)

        # --- Data Cleaning ---
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df.dropna(subset=['release_date'], inplace=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.fillna({'price': 0.0}, inplace=True)
        df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]

        # Connect and load data
        conn = sqlite3.connect(db_path)
        df.to_sql('steam_games', conn, if_exists='replace', index=False)
        print(f"Database '{db_path}' created and data loaded successfully.")
        conn.close()
    except FileNotFoundError:
        print(f"Error: The source file {csv_path} was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred during database setup: {e}")
        exit()

# --- 2. Natural Language to SQL Agent ---
class NlToSqlAgent:
    """
    An agent that translates natural language questions into SQL queries.
    """
    def __init__(self, csv_path='training_data.csv', db_path='steam_games.db'):
        """
        Initializes and trains the agent's classification model.
        """
        self.db_path = db_path
        
        try:
            # Use the csv module to correctly handle quoted fields
            questions = []
            queries = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip the header row
                for row in reader:
                    if len(row) == 2:
                        questions.append(row[0])
                        queries.append(row[1])
            
            self.questions = questions
            self.queries = queries

            # --- Model Training ---
            self.model = make_pipeline(
                TfidfVectorizer(),
                OneVsRestClassifier(LogisticRegression(solver='liblinear'))
            )
            self.model.fit(self.questions, self.queries)
            print("NL to SQL Agent (Direct Model) trained and ready.")
        except FileNotFoundError:
            print(f"Fatal Error: The training data file '{csv_path}' was not found.")
            raise

    def get_sql(self, natural_language_question: str) -> str:
        """
        Predicts a SQL query from a natural language question.
        """
        try:
            predicted_sql = self.model.predict([natural_language_question])[0]
            # Safeguard to ensure the correct table name is used
            if "FROM steam" in predicted_sql and "FROM steam_games" not in predicted_sql:
                 predicted_sql = predicted_sql.replace("FROM steam", "FROM steam_games")
            return predicted_sql
        except Exception as e:
            print(f"Error during SQL prediction: {e}")
            return None

    def interpret_and_execute(self, natural_language_question: str):
        """
        Takes a question, predicts SQL, executes it, and returns the results.
        """
        predicted_sql = self.get_sql(natural_language_question)
        if not predicted_sql:
            return None, "Failed to generate a SQL query."

        print(f"\nInterpreted question: '{natural_language_question}'")
        print(f"Predicted SQL Query: {predicted_sql}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(predicted_sql)
            result = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            conn.close()
            return column_names, result
        except sqlite3.Error as e:
            return None, f"An SQL error occurred: {e}"
        except Exception as e:
            return None, f"An unexpected error occurred during query execution: {e}"

# --- 3. Main Execution Block ---
def main():
    """
    Main function to run the agent in an interactive terminal session.
    """
    setup_database()

    try:
        agent = NlToSqlAgent()
    except Exception:
        print("\nCould not initialize agent. Exiting.")
        return

    print("\n--- Natural Language to SQL Agent ---")
    print("Ask questions about the Steam games dataset. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour question: > ").strip()
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue

        headers, data = agent.interpret_and_execute(user_input)
        
        if headers is None:
            print(f"Error: {data}")
        elif not data:
            print("--> The query executed successfully but returned no results.")
        else:
            df = pd.DataFrame(data, columns=headers)
            print("--> Query Result:")
            print(df.to_string(index=False))

if __name__ == '__main__':
    main()

