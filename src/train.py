import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
import time


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def build_model() -> OneVsRestClassifier:
    """Build and return a multi-class classifier using OneVsRestClassifier."""
    model = LogisticRegression(max_iter=1000, C=1.0, solver="saga")
    return OneVsRestClassifier(model)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> OneVsRestClassifier:
    """Train the model on provided training data."""
    print("Training model...")
    trained_model = build_model()
    trained_model.fit(X_train, y_train)
    print("Training complete.")
    return trained_model


def save_model(model: object, vectorizer: object, model_dir: str = "models/") -> None:
    """Save the trained model and vectorizer to disk."""
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f_model:
        pickle.dump(model, f_model)
    
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f_vectorizer:
        pickle.dump(vectorizer, f_vectorizer)


def load_model(model_dir: str = "models/") -> tuple:
    """Load the trained model and vectorizer from disk."""
    try:
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f_model:
            loaded_model = pickle.load(f_model)
        
        with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f_vectorizer:
            vectorizer = pickle.load(f_vectorizer)
            
        return (loaded_model, vectorizer)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model files not found in '{model_dir}'. "
            "Make sure the model has been trained and saved using save_model().\n" + str(e)
        )


def train_with_mlflow(X_train, X_test, y_train, y_test, experiment_name: str = "toxic-classifier") -> tuple:
    """Train model with MLflow tracking."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LogisticRegression-OneVsRest")
        mlflow.log_param("tfidf_max_features", 50000)
        mlflow.log_param("tfidf_ngram_range", "1,2")
        mlflow.log_param("lr_C", 1.0)
        mlflow.log_param("lr_max_iter", 1000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("lr_solver", "saga")

        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        scores = {}
        for i, label in enumerate(LABELS):
            score = f1_score(y_test.iloc[:, i], y_pred[:, i], average="binary", zero_division=0)
            scores[label] = score
            mlflow.log_metric(f"f1_{label}", score)
            print(f"F1 score for {label}: {score}")

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        mlflow.log_metric("macro_f1", macro_f1)
        print(f"Macro F1: {macro_f1}")

        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

    return (model, run_id)

