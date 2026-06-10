
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


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
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
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