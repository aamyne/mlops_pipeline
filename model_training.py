"""Module for training a Bagging model."""

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Any


def train_model(X_train: Any, y_train: Any, n_estimators: int = 10) -> BaggingClassifier:
    """
    Train a Bagging model with Decision Trees as the base estimator.

    Args:
        X_train: Training features (pandas DataFrame).
        y_train: Training labels (numpy array).
        n_estimators: Number of base estimators (default: 10).

    Returns:
        Trained BaggingClassifier model.
    """
    # One-hot encode categorical features for BaggingClassifier
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)

    # Define the Bagging model
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),  # Compatible with scikit-learn 1.2.0+
        n_estimators=n_estimators,
        random_state=42
    )

    # Train the model
    model.fit(X_train_encoded, y_train)

    # Log parameters and model to MLflow within the existing run
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("estimator", "DecisionTreeClassifier")
    mlflow.sklearn.log_model(model, "bagging_model")

    return model