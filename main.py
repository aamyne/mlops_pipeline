"""Main module for running the ML pipeline with enhanced MLflow tracking."""

import argparse
import sys
import time
import threading
import psutil
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import shap
from data_processing import prepare_data as process_data
from model_persistence import save_model

# Global variables to store prepared data
X_train, X_test, y_train, y_test = None, None, None, None
X_train_final_np, y_train_final_np, X_val_np, y_val_np, X_test_np, y_test_np = None, None, None, None, None, None

def prepare_data_wrapper(train_file: str, test_file: str) -> None:
    """Wrapper to prepare data and store it globally."""
    global X_train, X_test, y_train, y_test, X_train_final_np, y_train_final_np, X_val_np, y_val_np, X_test_np, y_test_np
    print("🔹 Preparing data...")
    X_train, X_test, y_train, y_test = process_data(train_file, test_file)
    X_val = X_train.sample(frac=0.2, random_state=42)
    y_val = y_train[X_val.index]
    X_train_final = X_train.drop(X_val.index)
    y_train_final = y_train[X_train_final.index]
    X_train_np = X_train.to_numpy()
    y_train_np = y_train
    X_train_final_np = X_train_final.to_numpy()
    y_train_final_np = y_train_final
    X_val_np = X_val.to_numpy()
    y_val_np = y_val
    X_test_np = X_test.to_numpy()
    y_test_np = y_test
    stats = X_train_final.describe().T
    stats.to_csv('dataset_stats.csv')
    mlflow.log_artifact('dataset_stats.csv')
    print("🔹 Data preparation complete")

def train_model_wrapper() -> None:
    """Wrapper to train the model using prepared data."""
    global X_train_final_np, y_train_final_np
    print("🔹 Training model...")
    process = psutil.Process()
    start_time = time.time()
    cpu_samples = []
    net_io_start = psutil.net_io_counters()

    params = {
        "n_estimators": 50,
        "random_state": 42,
        "estimator": DecisionTreeClassifier(max_depth=6)
    }
    mlflow.log_params({
        "n_estimators": params["n_estimators"],
        "random_state": params["random_state"],
        "max_depth": params["estimator"].max_depth
    })
    mlflow.log_param("model_version", "1.0.0")

    model = BaggingClassifier(**params)
    training_done = [False]
    def sample_resources():
        while not training_done[0]:
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
    cpu_thread = threading.Thread(target=sample_resources)
    cpu_thread.start()
    model.fit(X_train_final_np, y_train_final_np)
    training_done[0] = True
    cpu_thread.join()

    training_time = time.time() - start_time
    memory_usage_mb = process.memory_info().rss / 1024 / 1024
    cpu_usage = np.mean(cpu_samples) if cpu_samples else 0
    net_io_end = psutil.net_io_counters()
    network_transmit_mb = (net_io_end.bytes_sent - net_io_start.bytes_sent) / 1024 / 1024
    network_receive_mb = (net_io_end.bytes_recv - net_io_start.bytes_recv) / 1024 / 1024
    disk_usage = psutil.disk_usage('/')
    disk_usage_mb = disk_usage.used / 1024 / 1024
    disk_available_mb = disk_usage.free / 1024 / 1024
    disk_usage_percent = disk_usage.percent
    memory_usage_percent = (process.memory_info().rss / psutil.virtual_memory().total) * 100

    system_metrics = {
        "system/training_time_seconds": training_time,
        "system/system_memory_usage_megabytes": memory_usage_mb,
        "system/network_transmit_megabytes": network_transmit_mb,
        "system/disk_usage_megabytes": disk_usage_mb,
        "system/disk_available_megabytes": disk_available_mb,
        "system/disk_usage_percentage": disk_usage_percent,
        "system/cpu_utilization_percentage": cpu_usage,
        "system/network_receive_megabytes": network_receive_mb,
        "system/system_memory_usage_percentage": memory_usage_percent
    }
    mlflow.log_metrics(system_metrics)
    print(f"🔹 System Metrics: Training time: {training_time:.2f}s, Memory: {memory_usage_mb:.2f}MB, CPU: {cpu_usage:.2f}%")
    print("🔹 Model training complete")
    save_model(model, "model.joblib")
    mlflow.sklearn.log_model(model, "model")

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline."""
    with mlflow.start_run(run_name="Full_Pipeline"):
        prepare_data_wrapper(train_file, test_file)
        train_model_wrapper()
        # Add evaluation and plotting steps here if needed
        # For brevity, omitted full implementation

def main() -> None:
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline with Enhanced MLflow Tracking")
    parser.add_argument(
        "action",
        type=str,
        nargs="?",
        default="all",
        help="Action to perform: 'all' to run the complete pipeline."
    )
    args = parser.parse_args()

    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"

    try:
        if args.action == "all":
            run_full_pipeline(train_file, test_file)
        else:
            print("\n❌ Invalid action! Use 'all' to run the complete pipeline.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()