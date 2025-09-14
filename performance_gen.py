import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Import your model files ---
# Ensure 'direct_class.py' and 'template.py' are in the same folder.
try:
    from direct_class import NlToSqlAgent as DirectAgent
    from direct_class import setup_database
    from template import train_model as train_template_model
    from template import nl_to_sql as get_sql_from_template_model
except ImportError as e:
    print(f"Fatal Error: Could not import model files. Details: {e}")
    exit()

# --- Evaluation Configuration ---
DB_PATH = 'steam_games.db'
TRAINING_DATA_PATH = 'training_data.csv'
MODEL_PKL_PATH = "nl2sql_model.pkl"
CHART_DIRECTORY = "charts" # Directory to save the charts

TEST_SET = [
    {"question": "how many games are there in total", "ground_truth_sql": "SELECT COUNT(*) FROM steam_games;"},
    {"question": "List all games released after 2018", "ground_truth_sql": "SELECT * FROM steam_games WHERE release_date > '2018-12-31';"},
    {"question": "List all games released after 2019", "ground_truth_sql": "SELECT * FROM steam_games WHERE release_date > '2019-12-31';"},
    {"question": "find all action games", "ground_truth_sql": "SELECT * FROM steam_games WHERE genres LIKE '%Action%';"},
    {"question": "find all indie games", "ground_truth_sql": "SELECT * FROM steam_games WHERE genres LIKE '%Indie%';"},
    {"question": "show me games made by Valve", "ground_truth_sql": "SELECT * FROM steam_games WHERE developer LIKE '%Valve%';"}
]

# --- Helper Functions ---
def execute_query(sql_query, db_path):
    """Executes a SQL query and returns the results."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception:
        return None

# --- Chart Generation Function ---
def generate_accuracy_chart(scores, directory):
    """Generates and saves the Exact vs. Execution Match accuracy chart."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_labels = list(scores.keys())
    exact_match = [scores[model]['exact_match'] for model in model_labels]
    exec_match = [scores[model]['execution_match'] for model in model_labels]
    x = np.arange(len(model_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, exact_match, width, label='Exact Match', color='skyblue')
    rects2 = ax.bar(x + width/2, exec_match, width, label='Execution Match', color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance: Exact vs. Execution Match Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.set_ylim(0, 110)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    chart_path = os.path.join(directory, 'exact_vs_execution_chart.png')
    plt.savefig(chart_path)
    print(f"\nAccuracy chart saved to '{chart_path}'")

# --- Main Evaluation Logic ---
def run_evaluation():
    """Initializes models, runs the accuracy evaluation, and generates the chart."""
    print("Setting up the database for evaluation...")
    setup_database(db_path=DB_PATH)

    print("\nInitializing and training models...")
    try:
        direct_model = DirectAgent(csv_path=TRAINING_DATA_PATH, db_path=DB_PATH)
        if not os.path.exists(MODEL_PKL_PATH):
            train_template_model()
    except Exception as e:
        print(f"Failed to initialize models. Error: {e}")
        return

    # --- Accuracy Calculation ---
    scores = {"Direct SQL Model": {"exact_match": 0, "execution_match": 0}, "Template + Slot Model": {"exact_match": 0, "execution_match": 0}}
    for item in TEST_SET:
        gt_sql, question = item["ground_truth_sql"], item["question"]
        gt_results = execute_query(gt_sql, DB_PATH)
        
        # Evaluate Direct Model
        direct_sql = direct_model.get_sql(question)
        if direct_sql == gt_sql: scores["Direct SQL Model"]["exact_match"] += 1
        if execute_query(direct_sql, DB_PATH) == gt_results: scores["Direct SQL Model"]["execution_match"] += 1

        # Evaluate Template Model
        template_sql = get_sql_from_template_model(question)
        if template_sql == gt_sql: scores["Template + Slot Model"]["exact_match"] += 1
        if execute_query(template_sql, DB_PATH) == gt_results: scores["Template + Slot Model"]["execution_match"] += 1

    # Convert counts to percentages
    for model in scores:
        scores[model]["exact_match"] = (scores[model]["exact_match"] / len(TEST_SET)) * 100
        scores[model]["execution_match"] = (scores[model]["execution_match"] / len(TEST_SET)) * 100
    
    print("\nFinal Accuracy Scores:", scores)

    # --- Generate the Chart ---
    generate_accuracy_chart(scores, CHART_DIRECTORY)

if __name__ == '__main__':
    run_evaluation()