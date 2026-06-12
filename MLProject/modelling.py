import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import os
import warnings


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # =====================================================
    # LOAD DATASET
    # =====================================================

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "shopping_trends_preprocessing.csv"
    )

    data = pd.read_csv(file_path)

    # =====================================================
    # FEATURE & TARGET
    # =====================================================

    X = data.drop(columns=["Subscription Status"])
    y = data["Subscription Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    input_example = X_train.head(5)

    # =====================================================
    # BEST PARAMETER DARI TUNING
    # =====================================================

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )

    with mlflow.start_run():

        # TRAINING
        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        # =================================================
        # METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )

        # =================================================
        # PARAMETERS
        # =================================================

        mlflow.log_param(
            "n_estimators",
            50
        )

        mlflow.log_param(
            "max_depth",
            10
        )

        mlflow.log_param(
            "min_samples_split",
            2
        )

        # =================================================
        # METRICS
        # =================================================

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # =================================================
        # MODEL SIGNATURE
        # =================================================

        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # =================================================
        # LOG MODEL
        # =================================================

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print("\nTraining selesai")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")