"""Pipeline runner for ML model training and evaluation."""

from src.data_loader import load_raw_data, split_data
from src.preprocess import fit_transform_tfidf
from src.train import train_with_mlflow, save_model
from src.evaluate import evaluate_model, print_report

# Allow file store backend for MLflow
import os
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'


def run():
    """Run the complete ML pipeline."""
    print("Step 1: Loading data...")
    df = load_raw_data()
    print("Step 2: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    print("Step 3: Preprocessing and vectorizing...")
    X_train_tfidf, X_test_tfidf, vectorizer = fit_transform_tfidf(X_train, X_test)
    print("Step 4: Training model with MLflow tracking...")
    model, run_id = train_with_mlflow(X_train_tfidf, X_test_tfidf, y_train, y_test)
    print(f"MLflow run ID: {run_id}")
    print("Step 5: Saving model...")
    save_model(model, vectorizer)
    print("Step 6: Evaluating model...")
    evaluate_model(model, X_test_tfidf, y_test)
    print_report(model, X_test_tfidf, y_test)
    print("Pipeline complete.")


if __name__ == "__main__":
    run()



