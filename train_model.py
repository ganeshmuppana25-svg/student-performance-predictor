"""Train the student performance Decision Tree model.

Pipeline: load CSV -> validate -> clean (missing/duplicates/ranges)
-> stratified train/test split -> DecisionTreeClassifier -> evaluate
-> save model (model/student_model.pkl) + metrics (model/metrics.json).

Run:  python train_model.py
"""
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURES = [
    "attendance", "study_hours", "assignment_score", "previous_exam_score",
    "internal_test_score", "sleep_hours", "screen_time",
    "assignments_completed", "class_participation", "previous_backlogs",
]
CLASSES = ["Poor", "Average", "Good", "Excellent"]

# Allowed (min, max) range for each feature, used for validation + cleaning.
FEATURE_RANGES = {
    "attendance": (0.0, 100.0),
    "study_hours": (0.0, 12.0),
    "assignment_score": (0.0, 100.0),
    "previous_exam_score": (0.0, 100.0),
    "internal_test_score": (0.0, 100.0),
    "sleep_hours": (0.0, 12.0),
    "screen_time": (0.0, 16.0),
    "assignments_completed": (0.0, 100.0),
    "class_participation": (0.0, 100.0),
    "previous_backlogs": (0, 8),
}

RANDOM_STATE = 42


def load_and_validate(path):
    """Load CSV and validate structure, columns and target classes."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run generate_dataset.py first.")
    df = pd.read_csv(path)
    required = FEATURES + ["performance"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    if len(df) < 50:
        raise ValueError(f"Dataset too small ({len(df)} rows); at least 50 rows required.")

    unknown = [c for c in df.columns if c not in required]
    if unknown:
        print(f"Note: dropping extra column(s): {unknown}")
        df = df[required]

    df["performance"] = df["performance"].astype(str).str.strip().str.capitalize()
    bad_labels = set(df["performance"].unique()) - set(CLASSES)
    if bad_labels:
        print(f"Note: dropping rows with unknown labels: {bad_labels}")
        df = df[~df["performance"].isin(bad_labels)]
    if df.empty:
        raise ValueError("No valid labelled rows remain after validation.")
    if set(df["performance"].unique()) != set(CLASSES):
        raise ValueError("Dataset must contain all 4 classes: " + ", ".join(CLASSES))
    return df


def clean_data(df):
    """Handle missing values, duplicates and out-of-range values."""
    n0 = len(df)

    # Numeric coercion: invalid strings become NaN and are dropped.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before_missing = len(df)
    df = df.dropna(subset=FEATURES + ["performance"])
    dropped_invalid = before_missing - len(df)

    duplicates = int(df.duplicated(subset=FEATURES + ["performance"]).sum())
    df = df.drop_duplicates(subset=FEATURES + ["performance"]).reset_index(drop=True)

    # Clip out-of-range values into allowed ranges (report if any clipped).
    clipped = 0
    for col, (lo, hi) in FEATURE_RANGES.items():
        outside = ((df[col] < lo) | (df[col] > hi)).sum()
        clipped += int(outside)
        df[col] = df[col].clip(lo, hi)

    print(f"Cleaning: {n0} rows -> {len(df)} rows "
          f"(dropped invalid/missing: {dropped_invalid}, duplicates removed: {duplicates}, "
          f"values clipped: {clipped})")
    if len(df) < 50:
        raise ValueError("Too few valid rows remain after cleaning.")
    return df


def train():
    print("=" * 60)
    print("Student Performance Predictor - Model Training")
    print("=" * 60)

    df = load_and_validate(DATA_PATH)
    df = clean_data(df)
    print(f"Final dataset: {len(df)} rows")
    print(df["performance"].value_counts().to_string())

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["performance"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    print(f"DecisionTreeClassifier trained "
          f"(depth={clf.get_depth()}, leaves={clf.get_n_leaves()}).")

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, labels=CLASSES,
                                      average="macro", zero_division=0))
    recall = float(recall_score(y_test, y_pred, labels=CLASSES,
                                average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, labels=CLASSES, average="macro",
                        zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=CLASSES).tolist()
    print("\n--- Test-set performance (actual computed metrics) ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (macro)")
    print(f"Recall   : {recall:.4f} (macro)")
    print(f"F1 Score : {f1:.4f} (macro)")

    importances = {f: float(round(imp, 4))
                   for f, imp in sorted(zip(FEATURES, clf.feature_importances_),
                                        key=lambda kv: kv[1], reverse=True)}

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    metrics = {
        "model": "DecisionTreeClassifier",
        "trained_at": pd.Timestamp.now().isoformat(),
        "random_state": RANDOM_STATE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "tree_depth": int(clf.get_depth()),
        "tree_leaves": int(clf.get_n_leaves()),
        "classes": CLASSES,
        "features": FEATURES,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
        "dataset": {
            "total_rows": int(len(df)),
            "class_distribution": {c: int(n) for c, n in
                                   df["performance"].value_counts().items()},
            "feature_ranges": {k: list(v) for k, v in FEATURE_RANGES.items()},
        },
        "feature_importances": importances,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # Sanity check: reload and predict once to prove save/load round-trip.
    reloaded = joblib.load(MODEL_PATH)
    check = reloaded.predict(X_test[:3])
    assert list(check) == list(y_pred[:3]), "Reloaded model predictions differ!"
    print("Model reload sanity check passed.\n")
    return metrics


if __name__ == "__main__":
    train()

