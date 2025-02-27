"""Main module for running the ML pipeline with a Bagging model."""

import argparse
import mlflow
import sys
import os
from datetime import datetime
from data_processing import prepare_data
from model_training import train_model
from model_evaluation import evaluate_model
from model_persistence import save_model


def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "churn_prediction_bagging"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def get_model_version():
    """Generate model version based on date and time."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M")


def run_full_pipeline(train_file: str, test_file: str, n_estimators: int = 10) -> None:
    """Execute the complete ML pipeline with MLflow tracking."""
    print("Running full pipeline...")
    
    # Create artifacts directory
    os.makedirs("./artifacts", exist_ok=True)
    
    # Setup MLflow and get model version
    experiment_id = setup_mlflow()
    model_version = get_model_version()

    # Start a new MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Full_Pipeline_v{model_version}"):
        try:
            # Log input files and parameters
            mlflow.log_param("train_file", train_file)
            mlflow.log_param("test_file", test_file)
            mlflow.log_param("model_version", model_version)
            mlflow.log_param("n_estimators", n_estimators)

            # Data preparation
            print("🔹 Preparing data...")
            X_train, X_test, y_train, y_test = prepare_data(train_file, test_file)
            print("🔹 Data preparation complete")

            # Model training
            print("🔹 Training model...")
            model = train_model(X_train, y_train, n_estimators=n_estimators)
            print("🔹 Model training complete")

            # Model evaluation
            print("🔹 Evaluating model...")
            metrics = evaluate_model(model, X_test, y_test)
            print("🔹 Evaluation complete")

            # Save the model with a relative path
            save_model(model, filename=f"artifacts/model_v{model_version}.joblib")

            print(f"🔹 Pipeline completed successfully for version {model_version}")

        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise e


def main() -> None:
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline with Bagging")
    parser.add_argument(
        "action",
        type=str,
        nargs="?",
        default="all",
        help="Action to perform: 'all' to run the complete pipeline."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=10,
        help="Number of estimators for the Bagging model."
    )
    args = parser.parse_args()

    train_file = "churn-bigml-80.csv"
    test_file = "churn-bigml-20.csv"

    try:
        if args.action == "all":
            run_full_pipeline(train_file, test_file, n_estimators=args.n_estimators)
        else:
            print("\n❌ Invalid action! Choose 'all' to run the complete pipeline.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()