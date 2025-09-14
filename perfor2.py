import pandas as pd
import sqlite3
import os
import time
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

# --- Chart Generation Functions ---
def generate_accuracy_chart(scores, directory):
    model_labels = list(scores.keys())
    exact_match = [scores[model]['exact_match'] for model in model_labels]
    exec_match = [scores[model]['execution_match'] for model in model_labels]
    x = np.arange(len(model_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, exact_match, width, label='Exact Match', color='skyblue')
    rects2 = ax.bar(x + width/2, exec_match, width, label='Execution Match', color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance by Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.set_ylim(0, 110)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(os.path.join(directory, 'accuracy_chart.png'))
    print(f"Accuracy chart saved to '{directory}/accuracy_chart.png'")

def generate_latency_chart(latencies, directory):
    labels = list(latencies.keys())
    values = list(latencies.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color=['skyblue', 'coral'])
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Model Performance by Prediction Speed')

    for i, v in enumerate(values):
        ax.text(i, v + 0.05, f"{v:.2f} ms", ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(os.path.join(directory, 'latency_chart.png'))
    print(f"Latency chart saved to '{directory}/latency_chart.png'")

def generate_precision_recall_chart(scores, directory):
    # --- EDIT: Replaced Bar Chart with a Radar Chart ---
    labels = ['Precision (Exact)', 'Recall (Execution)', 'F1-Score']
    num_vars = len(labels)

    # Calculate angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot data for each model
    for i, (model_name, model_scores) in enumerate(scores.items()):
        precision = model_scores['exact_match']
        recall = model_scores['execution_match']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        values = [precision, recall, f1_score]
        values += values[:1] # Complete the loop
        
        color = 'skyblue' if i == 0 else 'coral'
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, color=color, alpha=0.25)

    # Formatting the chart
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Precision, Recall, and F1-Score Analysis', size=15, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig(os.path.join(directory, 'precision_recall_radar_chart.png'))
    print(f"Precision/Recall/F1 radar chart saved to '{directory}/precision_recall_radar_chart.png'")


# --- Main Evaluation Logic ---
def run_full_evaluation():
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
        
        direct_sql = direct_model.get_sql(question)
        if direct_sql == gt_sql: scores["Direct SQL Model"]["exact_match"] += 1
        if execute_query(direct_sql, DB_PATH) == gt_results: scores["Direct SQL Model"]["execution_match"] += 1

        template_sql = get_sql_from_template_model(question)
        if template_sql == gt_sql: scores["Template + Slot Model"]["exact_match"] += 1
        if execute_query(template_sql, DB_PATH) == gt_results: scores["Template + Slot Model"]["execution_match"] += 1

    for model in scores:
        scores[model]["exact_match"] = (scores[model]["exact_match"] / len(TEST_SET)) * 100
        scores[model]["execution_match"] = (scores[model]["execution_match"] / len(TEST_SET)) * 100
    
    print("\nFinal Accuracy Scores:", scores)

    # --- Latency Calculation ---
    latencies = {"Direct SQL Model": [], "Template + Slot Model": []}
    for _ in range(10):
        for item in TEST_SET:
            start_time = time.perf_counter()
            direct_model.get_sql(item["question"])
            latencies["Direct SQL Model"].append((time.perf_counter() - start_time) * 1000)
            
            start_time = time.perf_counter()
            get_sql_from_template_model(item["question"])
            latencies["Template + Slot Model"].append((time.perf_counter() - start_time) * 1000)
    
    avg_latencies = {model: np.mean(times) for model, times in latencies.items()}
    print("\nAverage Prediction Latency (ms):", avg_latencies)

    # --- Generate All Charts ---
    if not os.path.exists(CHART_DIRECTORY):
        os.makedirs(CHART_DIRECTORY)
    
    generate_accuracy_chart(scores, CHART_DIRECTORY)
    generate_latency_chart(avg_latencies, CHART_DIRECTORY)
    generate_precision_recall_chart(scores, CHART_DIRECTORY)

if __name__ == '__main__':
    run_full_evaluation()

