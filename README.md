# Student Performance Predictor

A web application that predicts a student's overall academic performance
(**Poor / Average / Good / Excellent**) from 10 academic and lifestyle
features, using a **scikit-learn Decision Tree** served through a **Flask API**.

**Stack:** HTML + CSS + JavaScript (frontend) · Python Flask (backend) ·
Scikit-learn DecisionTreeClassifier (ML) · CSV + Pandas (dataset).
No database, no external APIs, no deployment features.

## Flow

```
Student inputs → JavaScript → Flask API → Decision Tree
             → Prediction + Confidence → Frontend
```

## Features (exact 10 inputs)

| # | Feature | Range |
|---|---------|-------|
| 1 | Attendance (%) | 0–100 |
| 2 | Study Hours per Day | 0–12 |
| 3 | Assignment Score (%) | 0–100 |
| 4 | Previous Exam Score (%) | 0–100 |
| 5 | Internal/Test Score (%) | 0–100 |
| 6 | Sleep Hours | 0–12 |
| 7 | Daily Screen Time (hours) | 0–16 |
| 8 | Assignments Completed (%) | 0–100 |
| 9 | Class Participation (%) | 0–100 |
| 10 | Previous Backlogs | 0–8 |

Target classes: **Poor, Average, Good, Excellent** (4 classes).

## Project Structure

```
student-performance-predictor/
├── app.py                    # Flask backend + API endpoints
├── train_model.py            # Dataset validation, cleaning, training, metrics
├── generate_dataset.py       # One-time dataset generator
├── requirements.txt
├── README.md
├── data/student_performance.csv   # ~1200 rows, balanced 4-class target
├── model/student_model.pkl        # Trained Decision Tree (joblib)
├── model/metrics.json             # Actual computed metrics
├── templates/index.html
└── static/
    ├── style.css
    └── script.js
```

## Setup & Run

```bash
pip install -r requirements.txt

# 1. Generate the dataset (only needed once; dataset is included)
python generate_dataset.py

# 2. Train the model (creates model/student_model.pkl and model/metrics.json)
python train_model.py

# 3. Start the Flask app
python app.py
# then open http://127.0.0.1:5000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Frontend page |
| POST | `/api/predict` | 10 inputs (JSON) → prediction + confidence + probabilities + explanation |
| GET  | `/api/metrics` | Accuracy, Precision, Recall, F1 (computed on test set), dataset stats |
| GET  | `/api/dataset` | Full dataset as JSON |
| GET  | `/api/health` | Health check (model/metrics loaded status) |

### Example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance":92,"study_hours":6,"assignment_score":88,"previous_exam_score":85,
       "internal_test_score":82,"sleep_hours":7,"screen_time":3,
       "assignments_completed":95,"class_participation":80,"previous_backlogs":0}'
```

## Machine Learning Pipeline (`train_model.py`)

1. **Validation** — required columns present, ≥50 rows, exactly the 4 classes.
2. **Cleaning** — numeric coercion, missing/invalid rows dropped, duplicates removed, out-of-range values clipped.
3. **Split** — stratified 80/20 train/test.
4. **Model** — `DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)`.
5. **Metrics** — Accuracy, macro Precision/Recall/F1 + confusion matrix, **computed on the test set** (never hardcoded).
6. **Persistence** — joblib save + reload sanity check.

## UI Features (Phase 2)

- **Light/Dark mode** — header toggle, persisted in localStorage (`spp_theme`), applies to every section; dark theme uses deep navy with subtle blue glow (no plain black). Respects `prefers-reduced-motion`.
- **Modern dashboard** — hero section with badges, KPI cards for Accuracy/Precision/Recall/F1 (all fetched live from `/api/metrics`, never hardcoded), styled result card, dataset statistics with class-distribution bars.
- **Slider inputs** — all 10 inputs are slider + exact-number pairs kept in sync (same `id`, `min`, `max`, `step` as Phase 1); helper text under each.
- **Presets** — 🌟 Excellent Student, 📚 Average Student, ⚠️ At-Risk Student chips fill the existing inputs only (values within the allowed ranges).
- **Prediction experience** — spinner on the Predict button, smooth result reveal, confidence bar animates 0% → actual confidence, per-class coloring (Poor = red, Average = amber, Good = blue, Excellent = green).
- **History cards** — timestamp, prediction badge, confidence, "Restore inputs" per entry, Clear History; still stored under the `spp_history` localStorage key.
- **Responsive** — fluid grids, breakpoints at 960px/640px; sections stack and buttons stay usable on mobile; result card scrolls into view after prediction.

## UI Features (Phase 1, unchanged behavior)

- 10 validated input controls, Predict + Reset buttons
- Prediction result card with confidence bar and short explanation
- Input summary, Model Performance, Dataset Information, How It Works

Invalid input handling: client-side and server-side range/type checks; API/model
errors surface as friendly messages on the page.
