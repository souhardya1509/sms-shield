"""
Fast controlled experiment: duplicate-retaining vs deduplicated training.

Both models are evaluated on exactly the same unique validation/test messages.
Thresholds are selected using validation data only. Existing app/model files are
not overwritten.

Run from your project folder:
    python controlled_model_comparison.py

Dataset is searched in:
    data/spam.csv, spam.csv, spam(1).csv
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
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA_PATHS = [BASE / "data" / "spam.csv", BASE / "spam.csv", BASE / "spam(1).csv"]
TARGET_PRECISION = 0.95
SEED = 42


def load_data():
    path = next((p for p in DATA_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Dataset not found. Put the TSV file at data/spam.csv, spam.csv, "
            "or spam(1).csv beside this script."
        )
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"]) & df["message"].ne("")]
    return path, df


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", ComplementNB(alpha=0.5)),
    ])


def get_scores(y, scores, threshold):
    actual = (pd.Series(y).to_numpy() == "spam").astype(int)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, pred)),
        "precision": float(precision_score(actual, pred, zero_division=0)),
        "recall": float(recall_score(actual, pred, zero_division=0)),
        "f1": float(f1_score(actual, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_val, val_scores):
    candidates = [get_scores(y_val, val_scores, i / 100) for i in range(10, 91)]
    acceptable = [r for r in candidates if r["precision"] >= TARGET_PRECISION]
    if acceptable:
        return max(acceptable, key=lambda r: (r["recall"], r["f1"], r["precision"]))
    return get_scores(y_val, val_scores, 0.50)


def main():
    path, raw = load_data()
    raw_rows = len(raw)
    # Use first occurrence of each exact message for a stable unique split.
    unique = raw.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    duplicates_removed = raw_rows - len(unique)

    print(f"Dataset: {path}")
    print(f"Rows: {raw_rows}; exact duplicate rows removed for unique split: {duplicates_removed}")
    print(f"Unique messages: {len(unique)}")
    print(f"Unique labels: {unique['label'].value_counts().to_dict()}")

    X = unique["message"]
    y = unique["label"]
    X_train_unique, X_temp, y_train_unique, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # Keep the exact same unique test and validation sets for both variants.
    held_out_messages = set(X_val) | set(X_test)
    # Variant A: duplicate-retaining training, but never train on any held-out text.
    train_with_dups = raw[~raw["message"].isin(held_out_messages)].copy()
    # Variant B: one row per message in the unique training split.
    train_dedup = pd.DataFrame({"message": X_train_unique.to_numpy(), "label": y_train_unique.to_numpy()})

    variants = [
        ("A: duplicate-retaining training", train_with_dups),
        ("B: deduplicated training", train_dedup),
    ]

    report = {}
    for name, train_df in variants:
        model = build_model()
        model.fit(train_df["message"], train_df["label"])
        spam_index = list(model.classes_).index("spam")
        val_scores = model.predict_proba(X_val)[:, spam_index]
        test_scores = model.predict_proba(X_test)[:, spam_index]
        chosen = choose_threshold(y_val, val_scores)
        test_at_050 = get_scores(y_test, test_scores, 0.50)
        test_at_chosen = get_scores(y_test, test_scores, chosen["threshold"])

        print(f"\n{name}")
        print(f"Training rows: {len(train_df)}")
        print(f"Validation-selected spam threshold: {chosen['threshold']:.2f}")
        print(f"Validation precision/recall: {chosen['precision']*100:.2f}% / {chosen['recall']*100:.2f}%")
        print("Test at threshold 0.50:")
        print(f"  Accuracy {test_at_050['accuracy']*100:.2f}% | Precision {test_at_050['precision']*100:.2f}% | Recall {test_at_050['recall']*100:.2f}% | F1 {test_at_050['f1']*100:.2f}%")
        print(f"  TN={test_at_050['tn']} FP={test_at_050['fp']} FN={test_at_050['fn']} TP={test_at_050['tp']}")
        print(f"Test at selected threshold {chosen['threshold']:.2f}:")
        print(f"  Accuracy {test_at_chosen['accuracy']*100:.2f}% | Precision {test_at_chosen['precision']*100:.2f}% | Recall {test_at_chosen['recall']*100:.2f}% | F1 {test_at_chosen['f1']*100:.2f}%")
        print(f"  TN={test_at_chosen['tn']} FP={test_at_chosen['fp']} FN={test_at_chosen['fn']} TP={test_at_chosen['tp']}")

        report[name] = {
            "training_rows": int(len(train_df)),
            "validation_threshold": float(chosen["threshold"]),
            "validation_metrics": chosen,
            "test_at_050": test_at_050,
            "test_at_selected_threshold": test_at_chosen,
        }

        # Save only the deduplicated candidate for a possible later app update.
        if name.startswith("B:"):
            joblib.dump(model, BASE / "spam_model_controlled.pkl")
            (BASE / "spam_threshold_controlled.json").write_text(
                json.dumps({
                    "spam_threshold": float(chosen["threshold"]),
                    "target_validation_spam_precision": TARGET_PRECISION,
                    "note": "Threshold selected on validation data. NB scores may not be calibrated."
                }, indent=2),
                encoding="utf-8",
            )

    output = BASE / "controlled_comparison_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {output.name}")
    print("Saved deduplicated candidate: spam_model_controlled.pkl")
    print("Your current spam_model.pkl and app.py were not changed.")
    print("\nNote: the test set is the same unique held-out set for both variants.")


if __name__ == "__main__":
    main()
