"""Generate a realistic, reasonably balanced student performance dataset.

Creates data/student_performance.csv with the 10 input features + target label.
Run once before training:  python generate_dataset.py
"""
import csv
import os
import random

FEATURES = [
    "attendance", "study_hours", "assignment_score", "previous_exam_score",
    "internal_test_score", "sleep_hours", "screen_time",
    "assignments_completed", "class_participation", "previous_backlogs",
]
CLASSES = ["Poor", "Average", "Good", "Excellent"]
N_ROWS = 1200
random.seed(42)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def gen_profile(label_bias):
    """Generate one student's features with values biased by ability level."""
    bias = label_bias  # -2 (weak) .. +2 (strong)
    attendance = clamp(random.gauss(78 + bias * 6, 8), 35, 100)
    study_hours = clamp(random.gauss(3.0 + bias * 1.1, 1.2), 0.5, 10)
    assignment_score = clamp(random.gauss(68 + bias * 7, 9), 20, 100)
    previous_exam_score = clamp(random.gauss(64 + bias * 8, 10), 15, 100)
    internal_test_score = clamp(random.gauss(65 + bias * 7, 10), 10, 100)
    sleep_hours = clamp(random.gauss(7.0 + random.choice([-0.4, 0, 0.4]), 1.1), 3, 10)
    screen_time = clamp(random.gauss(5.0 - bias * 0.8, 1.5), 0.5, 12)
    assignments_completed = clamp(random.gauss(72 + bias * 8, 12), 10, 100)
    class_participation = clamp(random.gauss(58 + bias * 9, 14), 5, 100)
    previous_backlogs = clamp(round(random.gauss(1.5 - bias * 0.8, 1.1)), 0, 8)
    return {
        "attendance": round(attendance, 1),
        "study_hours": round(study_hours, 1),
        "assignment_score": round(assignment_score, 1),
        "previous_exam_score": round(previous_exam_score, 1),
        "internal_test_score": round(internal_test_score, 1),
        "sleep_hours": round(sleep_hours, 1),
        "screen_time": round(screen_time, 1),
        "assignments_completed": round(assignments_completed, 1),
        "class_participation": round(class_participation, 1),
        "previous_backlogs": previous_backlogs,
    }


def label_student(f):
    """Assign class from a weighted composite score (with noise)."""
    academic = (
        f["attendance"] * 0.15
        + f["study_hours"] * 4.5
        + f["assignment_score"] * 0.18
        + f["previous_exam_score"] * 0.18
        + f["internal_test_score"] * 0.15
        + f["assignments_completed"] * 0.12
        + f["class_participation"] * 0.07
    )
    penalty = f["screen_time"] * 1.5 + f["previous_backlogs"] * 4.0
    sleep_effect = 0 if 6.5 <= f["sleep_hours"] <= 8.5 else -6
    score = academic - penalty + sleep_effect + random.gauss(0, 6)
    # Thresholds calibrated to the composite score distribution so that
    # classes stay reasonably balanced (~25% each).
    if score < 38:
        return "Poor"
    if score < 58:
        return "Average"
    if score < 78:
        return "Good"
    return "Excellent"


def main():
    rows, counts = [], {c: 0 for c in CLASSES}
    # Roughly equal share of weak/strong students -> balanced classes
    biases = ([-2, -1, 0, 1, 2] * (N_ROWS // 5))[:N_ROWS]
    random.shuffle(biases)
    for bias in biases:
        f = gen_profile(bias)
        label = label_student(f)
        rows.append({**f, "performance": label})
        counts[label] += 1

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "student_performance.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURES + ["performance"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset written to {out_path} ({len(rows)} rows)")
    for c in CLASSES:
        print(f"  {c:<10} {counts[c]} ({counts[c] / len(rows):.1%})")


if __name__ == "__main__":
    main()
