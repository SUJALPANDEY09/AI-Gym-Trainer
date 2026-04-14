import csv
import json
import os
from datetime import datetime

LOG_CSV = "workout_log.csv"
LOG_JSON = "workout_log.json"


def save_workout(exercise, reps, accuracy=0, avg_rep_time=0.0, calories=0.0, duration_s=0):
    """Save workout session to both CSV and JSON logs."""
    timestamp = datetime.now().isoformat()
    row = {
        "exercise": exercise,
        "reps": reps,
        "accuracy": accuracy,
        "avg_rep_time": avg_rep_time,
        "calories": calories,
        "duration_s": round(duration_s),
        "timestamp": timestamp,
    }

    # --- CSV ---
    csv_exists = os.path.isfile(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

    # --- JSON ---
    history = []
    if os.path.isfile(LOG_JSON):
        try:
            with open(LOG_JSON, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []
    history.append(row)
    with open(LOG_JSON, "w") as f:
        json.dump(history, f, indent=2)

    return row


def load_history():
    """Load all workout history from JSON log."""
    if not os.path.isfile(LOG_JSON):
        return []
    try:
        with open(LOG_JSON, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_summary_stats(history=None):
    """Return aggregated statistics across all sessions."""
    if history is None:
        history = load_history()
    if not history:
        return {}

    by_exercise = {}
    for row in history:
        ex = row.get("exercise", "unknown")
        if ex not in by_exercise:
            by_exercise[ex] = {"sessions": 0, "total_reps": 0, "total_calories": 0.0, "accuracy_list": []}
        by_exercise[ex]["sessions"] += 1
        by_exercise[ex]["total_reps"] += row.get("reps", 0)
        by_exercise[ex]["total_calories"] += row.get("calories", 0.0)
        acc = row.get("accuracy", 0)
        if acc > 0:
            by_exercise[ex]["accuracy_list"].append(acc)

    summary = {}
    for ex, data in by_exercise.items():
        summary[ex] = {
            "sessions": data["sessions"],
            "total_reps": data["total_reps"],
            "total_calories": round(data["total_calories"], 1),
            "avg_accuracy": round(
                sum(data["accuracy_list"]) / len(data["accuracy_list"])
                if data["accuracy_list"] else 0
            ),
        }
    return summary
