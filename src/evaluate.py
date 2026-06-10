import numpy as np
from sklearn.metrics import classification_report, f1_score


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return dict with per-label F1 scores plus macro F1."""
    y_pred = model.predict(X_test)

    report_dict = {}
    
    for i, label in enumerate(LABELS):
        f1_single = f1_score(y_test[label], y_pred[:, i], average="binary")
        print(f"F1 score for {label}: {f1_single}")
        report_dict[label] = f1_single
    
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report_dict["macro_f1"] = macro_f1
    print(f"Macro F1: {macro_f1}")

    return report_dict


def print_report(model, X_test, y_test):
    """Print classification report for model."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=LABELS))