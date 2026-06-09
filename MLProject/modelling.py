import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import os
import sys
import warnings
import numpy as np


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # LOAD DATASET
    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "shopping_trends_preprocessing.csv"
        )
    )

    data = pd.read_csv(file_path)

    # FEATURE & TARGET
    X = data.drop(columns=["Subscription Status"])
    y = data["Subscription Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    input_example = X_train.head(5)

    # PARAMETER MODEL
    n_estimators = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 150
    )

    max_depth = (
        int(sys.argv[2])
        if len(sys.argv) > 2
        else 20
    )

    # MLFLOW RUN
    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        # Log parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"Accuracy : {accuracy:.4f}")