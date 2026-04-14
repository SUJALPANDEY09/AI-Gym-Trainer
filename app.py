"""
AI Gym Trainer — OpenCV standalone app
Controls:
  1-5  → switch exercise
  r    → reset reps
  s    → save & show summary
  q    → quit
"""

import cv2
import time
import numpy as np
from pose_module import PoseDetector
from exercise_logic import ExerciseTracker
from tracker import save_workout
from voice import speak, speak_rep_count, speak_feedback

EXERCISES = ["squat", "pushup", "bicep_curl", "lunge", "shoulder_press", "deadlift", "bench_press"]
EXERCISE_LABELS = {
    "squat": "Squat [1]",
    "pushup": "Push-up [2]",
    "bicep_curl": "Bicep Curl [3]",
    "lunge": "Lunge [4]",
    "shoulder_press": "Shoulder Press [5]",
}

# ─── Colors (BGR) ───────────────────────────────────────────────────
C_GREEN  = (80, 220, 80)
C_RED    = (60, 60, 220)
C_ORANGE = (30, 160, 240)
C_CYAN   = (220, 220, 40)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_PANEL  = (20, 20, 20)
C_ACCENT = (0, 200, 150)


def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.6):
    """Draw a filled semi-transparent rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_angle_arc(img, center, angle, color):
    """Draw a progress arc for the current joint angle."""
    x, y = center
    radius = 40
    thickness = 4
    pct = angle / 180.0
    end_angle = int(360 * pct)
    cv2.ellipse(img, (x, y), (radius, radius), -90, 0, 360, (60, 60, 60), thickness)
    cv2.ellipse(img, (x, y), (radius, radius), -90, 0, end_angle, color, thickness)
    cv2.putText(img, f"{int(angle)}", (x - 18, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, value, max_val, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * min(value / max_val, 1.0))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)


def draw_hud(img, exercise, tracker, stage, elapsed):
    h, w = img.shape[:2]

    # ── Left panel ────────────────────────────────────────────────
    draw_rounded_rect(img, 10, 10, 280, 340, 12, C_PANEL, alpha=0.65)

    # Title
    cv2.putText(img, "AI GYM TRAINER", (22, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C_ACCENT, 1, cv2.LINE_AA)

    # Exercise name
    ex_label = ExerciseTracker.EXERCISE_CONFIG.get(exercise, {}).get("name", exercise)
    cv2.putText(img, ex_label, (22, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WHITE, 2, cv2.LINE_AA)

    # Stage badge
    stage_color = C_GREEN if stage == "DOWN" else C_ORANGE
    cv2.rectangle(img, (22, 82), (140, 108), stage_color, -1, cv2.LINE_AA)
    cv2.putText(img, f"STAGE: {stage}", (28, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_BLACK, 2, cv2.LINE_AA)

    # Reps (big)
    cv2.putText(img, str(tracker.count), (22, 190),
                cv2.FONT_HERSHEY_DUPLEX, 4.5, C_GREEN, 6, cv2.LINE_AA)
    cv2.putText(img, "REPS", (22, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Accuracy bar
    cv2.putText(img, "ACCURACY", (22, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    acc_color = C_GREEN if tracker.accuracy >= 80 else (C_ORANGE if tracker.accuracy >= 50 else C_RED)
    draw_progress_bar(img, 22, 252, 236, 14, tracker.accuracy, 100, acc_color)
    cv2.putText(img, f"{tracker.accuracy}%", (220, 264),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

    # Calories
    cv2.putText(img, f"CALORIES: {tracker.calories_burned} kcal", (22, 295),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1, cv2.LINE_AA)

    # Timer
    mins, secs = divmod(int(elapsed), 60)
    cv2.putText(img, f"TIME: {mins:02d}:{secs:02d}", (22, 318),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    # Avg rep time
    cv2.putText(img, f"AVG REP: {tracker.avg_rep_time}s", (22, 342),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Angle arc (center bottom) ─────────────────────────────────
    cx = w // 2
    arc_color = acc_color
    draw_angle_arc(img, (cx, h - 60), tracker.current_angle, arc_color)
    cv2.putText(img, "ANGLE", (cx - 22, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Feedback banner (top center) ─────────────────────────────
    fb_color = C_GREEN if "Good" in tracker.feedback or "Great" in tracker.feedback else C_ORANGE
    if "Too" in tracker.feedback or "Ease" in tracker.feedback:
        fb_color = C_RED
    fb_text = tracker.feedback.upper()
    tw = cv2.getTextSize(fb_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
    bx = (w - tw - 24) // 2
    draw_rounded_rect(img, bx, h - 110, tw + 24, 36, 8, (20, 20, 20), 0.75)
    cv2.putText(img, fb_text, (bx + 12, h - 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fb_color, 2, cv2.LINE_AA)

    # ── Controls hint (bottom right) ─────────────────────────────
    hints = ["1-7: Switch Exercise  r: Reset  s: Save  q: Quit"]
    draw_rounded_rect(img, w - 310, h - 40, 300, 30, 6, C_PANEL, 0.5)
    cv2.putText(img, hints[0], (w - 304, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Per-rep accuracy sparkline (right panel) ──────────────────
    if tracker.per_rep_accuracy:
        draw_rounded_rect(img, w - 150, 10, 140, 160, 10, C_PANEL, 0.65)
        cv2.putText(img, "REP QUALITY", (w - 144, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        recent = tracker.per_rep_accuracy[-8:]
        bar_w = max(8, 120 // max(len(recent), 1))
        for i, acc in enumerate(recent):
            bh = int(acc * 0.8)
            bcolor = C_GREEN if acc >= 80 else (C_ORANGE if acc >= 50 else C_RED)
            bx2 = w - 144 + i * (bar_w + 3)
            cv2.rectangle(img, (bx2, 130 - bh), (bx2 + bar_w, 130), bcolor, -1)
        cv2.putText(img, f"AVG: {tracker.average_accuracy}%", (w - 144, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = PoseDetector(detection_confidence=0.7, tracking_confidence=0.7)
    tracker = ExerciseTracker()
    exercise = "squat"
    start_time = time.time()
    prev_count = 0

    speak("AI Gym Trainer started. Let's go!")

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        img = detector.find_pose(img)
        lmList = detector.find_position(img)

        if lmList:
            tracker.update(lmList, exercise)

            # Voice cues
            if tracker.count > prev_count:
                speak_rep_count(tracker.count)
                prev_count = tracker.count
            speak_feedback(tracker.feedback)

        elapsed = time.time() - start_time
        draw_hud(img, exercise, tracker, tracker.stage, elapsed)

        cv2.imshow("AI Gym Trainer", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            exercise = "squat"; tracker.reset(); prev_count = 0
        elif key == ord('2'):
            exercise = "pushup"; tracker.reset(); prev_count = 0
        elif key == ord('3'):
            exercise = "bicep_curl"; tracker.reset(); prev_count = 0
        elif key == ord('4'):
            exercise = "lunge"; tracker.reset(); prev_count = 0
        elif key == ord('5'):
            exercise = "shoulder_press"; tracker.reset(); prev_count = 0
        elif key == ord('6'):
            exercise = "deadlift"; tracker.reset(); prev_count = 0
        elif key == ord('7'):
            exercise = "bench_press"; tracker.reset(); prev_count = 0
        elif key == ord('r'):
            tracker.reset(); prev_count = 0
        elif key == ord('s'):
            save_workout(exercise, tracker.count, tracker.average_accuracy,
                         tracker.avg_rep_time, tracker.calories_burned,
                         int(time.time() - start_time))
            speak("Workout saved!")
        elif key == ord('q'):
            save_workout(exercise, tracker.count, tracker.average_accuracy,
                         tracker.avg_rep_time, tracker.calories_burned,
                         int(time.time() - start_time))
            speak("Great workout! Goodbye!")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
