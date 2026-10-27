"""
SMS Shield - Improved model training and evaluation

This script:
1. Loads the SMS Spam Collection TSV dataset.
2. Removes exact duplicate messages before splitting.
3. Creates stratified train/validation/test sets.
4. Trains the existing TF-IDF + Complement Naive Bayes pipeline.
5. Chooses a spam-probability threshold on the validation set,
   targeting at least 95% spam precision where possible.
6. Reports final test metrics at 0.50 and the selected threshold.
7. Saves a NEW model file and threshold file without overwriting
   the currently working app's model.

Run from your project folder:
    python train_improved_model.py

Expected dataset location:
    data/spam.csv
or:
    spam.csv
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

RANDOM_STATE = 42
TARGET_SPAM_PRECISION = 0.95

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "spam.csv",
    BASE_DIR / "spam.csv",
    BASE_DIR / "spam(1).csv",
]
MODEL_OUTPUT = BASE_DIR / "spam_model_improved.pkl"
THRESHOLD_OUTPUT = BASE_DIR / "spam_threshold.json"


def find_dataset():
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    candidates = "\n".join(f"  - {p}" for p in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the dataset. Put the SMS Spam Collection TSV file "
        "at one of these locations:\n" + candidates
    )


def load_dataset(path):
    # SMS Spam Collection is tab-separated and has no header.
    data = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    data = data.dropna(subset=["label", "message"]).copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"].ne("")]

    return data


def choose_threshold(y_true, spam_scores):
    """Maximize spam recall while meeting target precision on validation data."""
    y_binary = (pd.Series(y_true).to_numpy() == "spam").astype(int)
    best = None

    for threshold in [i / 100 for i in range(10, 91)]:
        predictions = (spam_scores >= threshold).astype(int)
        precision = precision_score(y_binary, predictions, zero_division=0)
        recall = recall_score(y_binary, predictions, zero_division=0)
        f1 = f1_score(y_binary, predictions, zero_division=0)

        if precision >= TARGET_SPAM_PRECISION:
            candidate = {
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            if best is None or (
                candidate["recall"],
                candidate["f1"],
                candidate["precision"],
            ) > (
                best["recall"],
                best["f1"],
                best["precision"],
            ):
                best = candidate

    # Fallback if no threshold meets the precision target.
    if best is None:
        best = {
            "threshold": 0.50,
            "precision": None,
            "recall": None,
            "f1": None,
        }
        print(
            "\nWARNING: No validation threshold reached "
            f"{TARGET_SPAM_PRECISION:.0%} spam precision. "
            "Using 0.50 as a fallback."
        )

    return best


def report_metrics(title, y_true, spam_scores, threshold):
    y_binary = (pd.Series(y_true).to_numpy() == "spam").astype(int)
    predictions = (spam_scores >= threshold).astype(int)

    print(f"\n--- {title} (spam threshold = {threshold:.2f}) ---")
    print(f"Accuracy:        {accuracy_score(y_binary, predictions) * 100:.2f}%")
    print(
        f"Spam precision:  "
        f"{precision_score(y_binary, predictions, zero_division=0) * 100:.2f}%"
    )
    print(
        f"Spam recall:     "
        f"{recall_score(y_binary, predictions, zero_division=0) * 100:.2f}%"
    )
    print(
        f"Spam F1-score:   "
        f"{f1_score(y_binary, predictions, zero_division=0) * 100:.2f}%"
    )
    print("Confusion matrix [[ham correct, ham flagged], [spam missed, spam caught]]:")
    print(confusion_matrix(y_binary, predictions, labels=[0, 1]))
    return predictions


def main():
    dataset_path = find_dataset()
    data = load_dataset(dataset_path)

    original_count = len(data)
    data = data.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    duplicates_removed = original_count - len(data)

    print(f"Dataset: {dataset_path}")
    print(f"Messages after basic validation: {original_count}")
    print(f"Exact duplicate messages removed: {duplicates_removed}")
    print(f"Unique messages used: {len(data)}")
    print("Label counts:")
    print(data["label"].value_counts())

    X = data["message"]
    y = data["label"]

    # 70% train, 15% validation, 15% final test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(
        f"\nSplit sizes — train: {len(X_train)}, "
        f"validation: {len(X_val)}, test: {len(X_test)}"
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", ComplementNB(alpha=0.5)),
        ]
    )

    print("\nTraining the TF-IDF + Complement Naive Bayes pipeline...")
    model.fit(X_train, y_train)

    classes = list(model.classes_)
    spam_index = classes.index("spam")

    # Threshold selection uses validation data only.
    validation_scores = model.predict_proba(X_val)[:, spam_index]
    selected = choose_threshold(y_val, validation_scores)

    print("\nSelected validation threshold:")
    print(f"Spam threshold: {selected['threshold']:.2f}")
    if selected["precision"] is not None:
        print(f"Validation spam precision: {selected['precision'] * 100:.2f}%")
        print(f"Validation spam recall:    {selected['recall'] * 100:.2f}%")

    # The final test set is used only for final reporting.
    test_scores = model.predict_proba(X_test)[:, spam_index]
    report_metrics("Final test results at default threshold", y_test, test_scores, 0.50)
    report_metrics(
        "Final test results at validation-selected threshold",
        y_test,
        test_scores,
        selected["threshold"],
    )

    joblib.dump(model, MODEL_OUTPUT)
    threshold_data = {
        "spam_threshold": selected["threshold"],
        "ham_threshold": 0.70,
        "target_validation_spam_precision": TARGET_SPAM_PRECISION,
        "dataset_unique_messages": int(len(data)),
        "duplicates_removed": int(duplicates_removed),
        "random_state": RANDOM_STATE,
        "note": (
            "Spam threshold was selected on validation data; ham threshold remains 0.70 for the app uncertainty band. Probability scores from "
            "Complement Naive Bayes are not guaranteed to be calibrated."
        ),
    }
    THRESHOLD_OUTPUT.write_text(json.dumps(threshold_data, indent=2), encoding="utf-8")

    print("\nSaved new model:", MODEL_OUTPUT.name)
    print("Saved threshold settings:", THRESHOLD_OUTPUT.name)
    print(
        "\nYour existing spam_model.pkl has NOT been overwritten. "
        "The current Streamlit app remains unchanged."
    )


if __name__ == "__main__":
    main()
