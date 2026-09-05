"""Student Performance Predictor - Flask backend.

Endpoints:
  GET  /                 -> frontend page
  POST /api/predict      -> 10 inputs -> prediction + confidence
  GET  /api/metrics      -> model metrics (accuracy/precision/recall/f1)
  GET  /api/dataset      -> dataset rows as JSON
  GET  /api/health       -> health check
"""
import json
import os
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")

# Exact feature order expected by the trained model.
FEATURES = [
    ("attendance", "Attendance (%)", 0.0, 100.0, "percent"),
    ("study_hours", "Study Hours per Day", 0.0, 12.0, "hours"),
    ("assignment_score", "Assignment Score (%)", 0.0, 100.0, "percent"),
    ("previous_exam_score", "Previous Exam Score (%)", 0.0, 100.0, "percent"),
    ("internal_test_score", "Internal/Test Score (%)", 0.0, 100.0, "percent"),
    ("sleep_hours", "Sleep Hours", 0.0, 12.0, "hours"),
    ("screen_time", "Daily Screen Time (hours)", 0.0, 16.0, "hours"),
    ("assignments_completed", "Assignments Completed (%)", 0.0, 100.0, "percent"),
    ("class_participation", "Class Participation (%)", 0.0, 100.0, "percent"),
    ("previous_backlogs", "Previous Backlogs", 0, 8, "count"),
]

CLASSES = ["Poor", "Average", "Good", "Excellent"]

app = Flask(__name__)
MODEL = None
METRICS = None


def load_model():
    global MODEL, METRICS
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    MODEL = joblib.load(MODEL_PATH)
    METRICS = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as fh:
            METRICS = json.load(fh)


def validate_inputs(payload):
    """Validate and convert request JSON to the model's feature vector.

    Returns (values_dict, error_message). error_message is None when valid.
    """
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    values, errors = {}, []
    for key, label, lo, hi, _unit in FEATURES:
        if key not in payload or payload[key] in (None, ""):
            errors.append(f"'{label}' is required.")
            continue
        raw = payload[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            errors.append(f"'{label}' must be a number.")
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{label}' must be a valid number.")
            continue
        if num != num:  # NaN
            errors.append(f"'{label}' must be a valid number.")
            continue
        if not (lo <= num <= hi):
            errors.append(f"'{label}' must be between {lo:g} and {hi:g}.")
            continue
        values[key] = num

    if errors:
        return None, "Invalid input: " + " ".join(errors)
    return values, None


def build_explanation(values, prediction, confidence):
    """Short rule-based explanation of what drove the prediction."""
    drivers = []
    if values["attendance"] >= 85:
        drivers.append("strong attendance")
    elif values["attendance"] < 65:
        drivers.append("low attendance")
    if values["study_hours"] >= 5:
        drivers.append("high study time")
    elif values["study_hours"] < 2:
        drivers.append("limited study time")
    if values["assignment_score"] >= 80 and values["assignments_completed"] >= 85:
        drivers.append("consistent assignment work")
    if values["previous_exam_score"] >= 75:
        drivers.append("solid previous exam results")
    if values["previous_backlogs"] >= 3:
        drivers.append("multiple previous backlogs")
    if values["screen_time"] >= 8:
        drivers.append("high daily screen time")
    if 6.5 <= values["sleep_hours"] <= 8.5:
        drivers.append("healthy sleep pattern")
    if not drivers:
        drivers.append("moderate academic habits overall")

    conf_word = ("high" if confidence >= 70 else
                 "moderate" if confidence >= 50 else "low")
    return (f"Predicted {prediction} performance with {conf_word} confidence. "
            f"Key factors: {', '.join(drivers[:4])}.")



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model is not loaded. Train the model first "
                                 "(run train_model.py) and restart the app."}), 503
    try:
        payload = request.get_json(silent=True)
        values, err = validate_inputs(payload)
        if err:
            return jsonify({"error": err}), 400

        import numpy as np
        row = [values[key] for key, *_ in FEATURES]
        proba = MODEL.predict_proba([row])[0]
        classes = list(MODEL.classes_)
        idx = int(proba.argmax())
        prediction = classes[idx]
        confidence = round(float(proba[idx]) * 100, 1)
        probabilities = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "explanation": build_explanation(values, prediction, confidence),
            "input_summary": values,
        })
    except Exception:
        app.logger.error("Prediction failed:\n%s", traceback.format_exc())
        return jsonify({"error": "Internal error while predicting. Please try again."}), 500


@app.route("/api/metrics")
def metrics():
    if METRICS is None:
        return jsonify({"error": "Metrics not found. Run train_model.py first."}), 503
    return jsonify(METRICS)


@app.route("/api/dataset")
def dataset():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset not found. Run generate_dataset.py first."}), 503
    try:
        df = pd.read_csv(DATASET_PATH)
        return jsonify({
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        })
    except Exception:
        app.logger.error("Failed to read dataset:\n%s", traceback.format_exc())
        return jsonify({"error": "Failed to load dataset."}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "metrics_loaded": METRICS is not None,
        "classes": CLASSES,
        "features": [f[0] for f in FEATURES],
    })


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
