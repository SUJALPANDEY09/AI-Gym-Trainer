# 🏋️ AI Gym Trainer

A **computer vision-powered fitness assistant** using MediaPipe Pose for real-time exercise tracking, form analysis, and workout analytics.

---

## ✨ Features

### Core
- **5 Exercises**: Squat, Push-up, Bicep Curl, Lunge, Shoulder Press
- **Real-time rep counting** with joint angle state machine
- **Form accuracy scoring** per rep (deviation from ideal joint angle)
- **Live voice coaching** — non-blocking TTS with cooldown
- **Landmark smoothing** — rolling average over 5 frames for stable tracking

### OpenCV App (`app.py`)
- Full-screen HUD with semi-transparent panels
- Animated angle arc display (colored by accuracy)
- Per-rep accuracy sparkline chart
- Stage badge, feedback banner, calorie counter
- Keyboard controls: `1–5` switch exercise, `r` reset, `s` save, `q` quit

### Streamlit Web App (`web_app.py`)
- Live camera feed with overlay
- Giant rep counter, angle readout, stage badge
- Color-coded form feedback banner (green / amber / red)
- Real-time angle line chart with 120-frame rolling window
- Full **History & Analytics** tab:
  - Lifetime stats cards (reps, calories, accuracy, sessions)
  - Reps stacked bar chart by exercise and date
  - Accuracy trend line chart
  - Per-exercise breakdown cards
  - Recent sessions table

### Data & Logging
- Dual logging: **CSV** and **JSON**
- Per-session: reps, accuracy, avg rep time, calories, duration

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

### Run the OpenCV App
```bash
python app.py
```

### Run the Streamlit Web App
```bash
streamlit run web_app.py
```

---

## 🧠 Architecture

```
app.py / web_app.py   ← Entry points
    │
    ├── pose_module.py   ← MediaPipe Pose + landmark smoothing
    ├── exercise_logic.py← State machine, rep counting, accuracy
    ├── utils.py         ← Angle math (signed joint angle)
    ├── tracker.py       ← CSV + JSON logging, history, stats
    └── voice.py         ← Non-blocking TTS with cooldown
```

---

## 🎮 Controls (OpenCV App)

| Key | Action |
|-----|--------|
| `1` | Switch to Squat |
| `2` | Switch to Push-up |
| `3` | Switch to Bicep Curl |
| `4` | Switch to Lunge |
| `5` | Switch to Shoulder Press |
| `r` | Reset rep counter |
| `s` | Save current session |
| `q` | Save and quit |

---

## 📐 How Accuracy is Scored

Each rep's accuracy = `max(0, 100 - deviation × 1.5)` where *deviation* is the absolute difference between the deepest angle achieved and the ideal target angle (90° for most exercises). A perfect deep squat/pushup scores 100%.
=======
# AI-Gym-Trainer
AI-powered Gym Trainer using Computer Vision for pose detection, rep counting, and posture correction
>>>>>>> 6e391174e9f3140c5486de71bb2eaa2130021d2b
