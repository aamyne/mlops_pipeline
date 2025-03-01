"""Main module for running the ML pipeline with enhanced MLflow tracking in a single run."""

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

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with enhanced MLflow tracking in a single run."""
    print("Running full pipeline...")

    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "churn_prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Start a single MLflow run
    with mlflow.start_run(run_name="Enhanced_Pipeline_With_System_Metrics"):
        try:
            # Log input files
            mlflow.log_param("train_file", train_file)
            mlflow.log_param("test_file", test_file)

            # Job 3: Prepare the data
            print("🔹 Preparing data...")
            X_train, X_test, y_train, y_test = process_data(train_file, test_file)
            X_val = X_train.sample(frac=0.2, random_state=42)
            y_val = y_train[X_val.index]
            X_train_final = X_train.drop(X_val.index)
            y_train_final = y_train[X_train_final.index]

            # Convert DataFrames to NumPy arrays
            X_train_np = X_train.to_numpy()
            y_train_np = y_train
            X_train_final_np = X_train_final.to_numpy()
            y_train_final_np = y_train_final
            X_val_np = X_val.to_numpy()
            y_val_np = y_val
            X_test_np = X_test.to_numpy()
            y_test_np = y_test

            # Dataset Statistics Summary
            stats = X_train_final.describe().T
            stats.to_csv('dataset_stats.csv')
            mlflow.log_artifact('dataset_stats.csv', artifact_path="data_stats")
            print("🔹 Data preparation complete")

            # Job 4: Train the model
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

            # Feature Importance Plot
            feature_importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
            feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(len(feature_importances))]
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, feature_importances)
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Plot')
            plt.savefig('feature_importance.png')
            plt.close()
            mlflow.log_artifact('feature_importance.png')

            # Validation Metrics vs. Number of Estimators
            val_accuracies = []
            val_loglosses = []
            for k in range(1, params['n_estimators'] + 1):
                estimators = model.estimators_[:k]
                pred_proba = np.mean([est.predict_proba(X_val_np) for est in estimators], axis=0)
                pred = np.argmax(pred_proba, axis=1)
                acc = accuracy_score(y_val_np, pred)
                ll = log_loss(y_val_np, pred_proba)
                val_accuracies.append(acc)
                val_loglosses.append(ll)

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, params['n_estimators'] + 1), val_accuracies, label='Validation Accuracy')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy vs. Number of Estimators')
            plt.legend()
            plt.savefig('accuracy_vs_estimators.png')
            plt.close()
            mlflow.log_artifact('accuracy_vs_estimators.png')

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, params['n_estimators'] + 1), val_loglosses, label='Validation Log Loss')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Log Loss')
            plt.title('Validation Log Loss vs. Number of Estimators')
            plt.legend()
            plt.savefig('logloss_vs_estimators.png')
            plt.close()
            mlflow.log_artifact('logloss_vs_estimators.png')

            # Learning Curve Plot
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train_np, y_train_np, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
            )
            train_mean = np.mean(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, label='Training Accuracy')
            plt.plot(train_sizes, val_mean, label='Validation Accuracy')
            plt.xlabel('Training Size')
            plt.ylabel('Accuracy')
            plt.title('Learning Curve')
            plt.legend()
            plt.savefig('learning_curve.png')
            plt.close()
            mlflow.log_artifact('learning_curve.png')

            # SHAP Summary Plot
            explainer = shap.TreeExplainer(model.estimators_[0])
            shap_values = explainer.shap_values(X_test_np)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            plt.savefig('shap_summary.png')
            plt.close()
            mlflow.log_artifact('shap_summary.png')

            # Evaluate the model
            print("🔹 Evaluating model...")
            y_pred = model.predict(X_test_np)
            y_pred_proba = model.predict_proba(X_test_np)

            # Log evaluation metrics
            metrics = {
                "test_accuracy": accuracy_score(y_test_np, y_pred),
                "test_log_loss": log_loss(y_test_np, y_pred_proba)
            }
            mlflow.log_metrics(metrics)
            print(f"🔹 Evaluation metrics: {metrics}")

            # Confusion Matrix Plot
            cm = confusion_matrix(y_test_np, y_pred)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.close()
            mlflow.log_artifact('confusion_matrix.png')

            # ROC Curve Plot
            fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig('roc_curve.png')
            plt.close()
            mlflow.log_artifact('roc_curve.png')

            # Test Predictions CSV
            pred_df = pd.DataFrame({
                'true_label': y_test_np,
                'predicted_label': y_pred,
                'predicted_prob_0': y_pred_proba[:, 0],
                'predicted_prob_1': y_pred_proba[:, 1]
            })
            pred_df.to_csv('test_predictions.csv', index=False)
            mlflow.log_artifact('test_predictions.csv')

            # Save and log the model with input example
            save_model(model, "model.joblib")
            input_example = X_train_final_np[:1]  # Use first row as example
            mlflow.sklearn.log_model(model, "model", input_example=input_example)

        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise e

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