# template.py
# This file contains the implementation of a template + slot-filling classification model

import pandas as pd
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import re

# -------------------------
# 1. Database Configuration
# -------------------------
DB_FILE = "steam_games.db"
CSV_FILE = "steam.csv"
TRAIN_FILE = "train_nl_sql.csv"

# -------------------------
# 2. Training Data
# -------------------------
# This section creates a default training file if one doesn't exist.
if not os.path.exists(TRAIN_FILE):
    train_data = [
        # Note: The SQL templates use placeholders like '%genre%' or > 'year'
        # which will be filled by the nl_to_sql function.
        ("how many games are there", "SELECT COUNT(*) FROM steam_games;", ""),
        ("show me games made by Valve", "SELECT * FROM steam_games WHERE developer LIKE '%developer%';", "developer"),
        ("what are the top 5 most expensive games", "SELECT * FROM steam_games ORDER BY price DESC LIMIT 5;", ""),
        ("list all games released after 2018", "SELECT * FROM steam_games WHERE release_date > 'year-12-31';", "year"),
        ("find all action games", "SELECT * FROM steam_games WHERE genres LIKE '%genre%';", "genre"),
        ("which games have the most positive ratings", "SELECT * FROM steam_games ORDER BY positive_ratings DESC LIMIT 10;", ""),
        ("show me all the free to play games", "SELECT * FROM steam_games WHERE price = 0;", ""),
        ("list all single player games", "SELECT * FROM steam_games WHERE categories LIKE '%mode%';", "mode"),
        ("list total price of all games", "SELECT SUM(price) FROM steam_games;", ""),
        ("games developed by Gearbox Software", "SELECT * FROM steam_games WHERE developer LIKE '%developer%';", "developer"),
        ("games released after 2020", "SELECT * FROM steam_games WHERE release_date > 'year-12-31';", "year"),
        ("find all adventure games", "SELECT * FROM steam_games WHERE genres LIKE '%genre%';", "genre"),
        ("list multiplayer games", "SELECT * FROM steam_games WHERE categories LIKE '%mode%';", "mode"),
    ]
    pd.DataFrame(train_data, columns=["nl", "sql", "slots"]).to_csv(TRAIN_FILE, index=False)
    print(f"Wrote default training CSV to {TRAIN_FILE}")

# -------------------------
# 3. Train classifier
# -------------------------
def train_model():
    """
    Trains the classifier and saves it to a file.
    """
    print("Training NL->template classifier...")
    train_df = pd.read_csv(TRAIN_FILE)
    X = train_df["nl"]
    y = train_df.index  # predict row index of template

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    joblib.dump((pipeline, train_df), "nl2sql_model.pkl")
    print(f"Trained NL->template classifier on {len(train_df)} examples and saved to nl2sql_model.pkl.")

# -------------------------
# 4. NL -> SQL (with improved slot-filling)
# -------------------------
def nl_to_sql(nl: str):
    """
    Takes a natural language query, predicts a SQL template, and fills in the slots.
    """
    pipeline, train_df = joblib.load("nl2sql_model.pkl")
    pred_idx = pipeline.predict([nl])[0]
    tpl_entry = train_df.iloc[pred_idx]
    sql = tpl_entry["sql"]
    slots = str(tpl_entry["slots"])

    subs = {}
    
    # Improved logic for extracting parameters (slots)
    if "year" in slots:
        match = re.search(r'\b(19|20)\d{2}\b', nl)
        if match:
            subs["year"] = match.group(0)

    if "genre" in slots:
        # This list can be expanded for more genres
        for genre in ["Action", "Adventure", "RPG", "Indie", "Strategy", "Simulation", "Casual"]:
            if genre.lower() in nl.lower():
                subs["genre"] = genre
                break 

    if "developer" in slots:
        # A better heuristic to find capitalized names that are not the first word.
        words = nl.split()
        candidates = [word for i, word in enumerate(words) if i > 0 and word[0].isupper()]
        if candidates:
            subs["developer"] = " ".join(candidates)

    if "mode" in slots:
        if "multiplayer" in nl.lower() or "multi-player" in nl.lower():
            subs["mode"] = "Multi-player"
        elif "singleplayer" in nl.lower() or "single-player" in nl.lower():
            subs["mode"] = "Single-player"

    # Fill the slots in the SQL template
    for key, val in subs.items():
        # This handles different ways the placeholder might appear in the SQL template
        sql = sql.replace(f"'{key}-12-31'", f"'{val}-12-31'")
        sql = sql.replace(f"%{key}%", f"%{val}%")
        sql = sql.replace(f"'{key}'", f"'{val}'")

    return sql

# -------------------------
# 5. REPL (for standalone testing)
# -------------------------
def main():
    print("\nNatural Language -> SQL agent (type 'exit' or Ctrl-C to quit)")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    while True:
        try:
            nl = input("\nNL> ").strip()
            if nl.lower() in ("exit", "quit"):
                break
            sql = nl_to_sql(nl)
            print(f"SQL> {sql}")
            try:
                res = cur.execute(sql).fetchall()
                if res:
                    # Print results in a more readable format
                    df = pd.DataFrame(res, columns=[desc[0] for desc in cur.description])
                    print("Result:")
                    print(df.to_string(index=False))
                else:
                    print("Query returned no results.")
            except Exception as e:
                print("SQL execution error:", e)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)

    conn.close()

if __name__ == "__main__":
    # If the model hasn't been trained yet, train it first.
    if not os.path.exists("nl2sql_model.pkl"):
        train_model()
    main()