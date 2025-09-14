import pandas as pd
import sqlite3
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Import your model files ---
# Make sure these files are in the same directory
try:
    from direct_class import NlToSqlAgent as DirectAgent
    from direct_class import setup_database
    from template import train_model as train_template_model
    from template import nl_to_sql as get_sql_from_template_model
except ImportError as e:
    print(f"Fatal Error: Could not import model files. Make sure 'direct_class.py' and 'template.py' are in the same folder. Details: {e}")
    exit()

# --- Evaluation Setup ---
DB_PATH = 'steam_games.db'
TRAINING_DATA_PATH = 'training_data.csv'
MODEL_PKL_PATH = "nl2sql_model.pkl"

# A comprehensive test set to evaluate both models
TEST_SET = [
    {
        "question": "how many games are there in total",
        "ground_truth_sql": "SELECT COUNT(*) FROM steam_games;"
    },
    {
        "question": "List all games released after 2018",
        "ground_truth_sql": "SELECT * FROM steam_games WHERE release_date > '2018-12-31';"
    },
    {
        "question": "List all games released after 2019", # Unseen parameter for Direct model
        "ground_truth_sql": "SELECT * FROM steam_games WHERE release_date > '2019-12-31';"
    },
    {
        "question": "find all action games",
        "ground_truth_sql": "SELECT * FROM steam_games WHERE genres LIKE '%Action%';"
    },
    {
        "question": "find all indie games", # Unseen parameter for Direct model
        "ground_truth_sql": "SELECT * FROM steam_games WHERE genres LIKE '%Indie%';"
    },
    {
        "question": "show me games made by Valve",
        "ground_truth_sql": "SELECT * FROM steam_games WHERE developer LIKE '%Valve%';"
    }
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
        # Return a unique error object if execution fails
        return None

# --- Main Evaluation Logic ---
def run_evaluation():
    """
    Runs the full evaluation suite: accuracy tests and latency tests.
    """
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

    # --- Accuracy Evaluation ---
    print("\n--- Starting Model Accuracy Evaluation ---")
    scores = {
        "Direct SQL Model": {"exact_match": 0, "execution_match": 0},
        "Template + Slot Model": {"exact_match": 0, "execution_match": 0}
    }

    for item in TEST_SET:
        question = item["question"]
        gt_sql = item["ground_truth_sql"]
        print(f"\n--- Evaluating Question: '{question}' ---")
        print(f"[Ground Truth SQL] {gt_sql}")

        gt_results = execute_query(gt_sql, DB_PATH)

        # Evaluate Direct Model
        direct_sql = direct_model.get_sql(question)
        print(f"[Direct Model SQL]   {direct_sql}")
        if direct_sql == gt_sql:
            scores["Direct SQL Model"]["exact_match"] += 1
            print("    -> Exact Match: PASS")
        else:
            print("    -> Exact Match: FAIL")
        
        direct_results = execute_query(direct_sql, DB_PATH)
        if direct_results is not None and direct_results == gt_results:
            scores["Direct SQL Model"]["execution_match"] += 1
            print("    -> Execution Match: PASS")
        else:
            print("    -> Execution Match: FAIL")

        # Evaluate Template Model
        template_sql = get_sql_from_template_model(question)
        print(f"[Template Model SQL] {template_sql}")
        if template_sql == gt_sql:
            scores["Template + Slot Model"]["exact_match"] += 1
            print("    -> Exact Match: PASS")
        else:
            print("    -> Exact Match: FAIL")
            
        template_results = execute_query(template_sql, DB_PATH)
        if template_results is not None and template_results == gt_results:
            scores["Template + Slot Model"]["execution_match"] += 1
            print("    -> Execution Match: PASS")
        else:
            print("    -> Execution Match: FAIL")

    # Convert counts to percentages
    num_questions = len(TEST_SET)
    for model_name in scores:
        for metric in scores[model_name]:
            scores[model_name][metric] = (scores[model_name][metric] / num_questions) * 100
    
    print("\n-----------------------------------")
    print("\nFinal Accuracy Scores:")
    print(scores)

    # --- Latency Evaluation ---
    print("\n--- Starting Model Latency Evaluation ---")
    latencies = {
        "Direct SQL Model": [],
        "Template + Slot Model": []
    }
    # Run multiple iterations to get a stable average
    for _ in range(10):
        for item in TEST_SET:
            question = item["question"]
            
            start_time = time.perf_counter()
            direct_model.get_sql(question)
            end_time = time.perf_counter()
            latencies["Direct SQL Model"].append((end_time - start_time) * 1000) # milliseconds

            start_time = time.perf_counter()
            get_sql_from_template_model(question)
            end_time = time.perf_counter()
            latencies["Template + Slot Model"].append((end_time - start_time) * 1000)

    avg_latencies = {
        "Direct SQL Model": np.mean(latencies["Direct SQL Model"]),
        "Template + Slot Model": np.mean(latencies["Template + Slot Model"])
    }
    print("\nAverage Prediction Latency (ms):")
    print(avg_latencies)

    # --- Generate Charts ---
    generate_charts(scores, avg_latencies)

def generate_charts(scores, avg_latencies):
    """
    Generates and saves bar charts for accuracy and latency.
    """
    # Accuracy Chart
    labels = list(scores.keys())
    exact_match_scores = [scores[model]['exact_match'] for model in labels]
    exec_match_scores = [scores[model]['execution_match'] for model in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    rects1 = ax1.bar(x - width/2, exact_match_scores, width, label='Exact Match', color='skyblue')
    rects2 = ax1.bar(x + width/2, exec_match_scores, width, label='Execution Match', color='coral')

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Performance by Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.set_ylim(0, 110)

    # Add labels to the accuracy bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Latency Chart
    latency_labels = list(avg_latencies.keys())
    latency_values = list(avg_latencies.values())
    
    ax2.bar(latency_labels, latency_values, color=['skyblue', 'coral'])
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('Model Performance by Prediction Speed')

    # Add labels to the latency bars
    for i, v in enumerate(latency_values):
        ax2.text(i, v + 0.1, f"{v:.2f} ms", ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig('model_performance_charts.png')
    print("\nCharts saved to model_performance_charts.png")


if __name__ == '__main__':
    run_evaluation()
