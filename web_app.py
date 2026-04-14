"""
AI Gym Trainer — Streamlit Web App
Run with: streamlit run web_app.py
"""

import streamlit as st
import cv2
import time
import numpy as np
from datetime import datetime
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from pose_module import PoseDetector
from exercise_logic import ExerciseTracker
from tracker import save_workout, load_history, get_summary_stats
from voice import speak, speak_rep_count

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Gym Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark theme overlay */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1320 50%, #0a1018 100%);
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Custom metric cards */
  .metric-card {
    background: linear-gradient(145deg, #12192b, #0f1624);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 200, 150, 0.15);
  }
  .metric-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #5a7fa0;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 42px;
    font-weight: 700;
    color: #00e6a0;
    line-height: 1;
  }
  .metric-sub {
    font-size: 13px;
    color: #4a6a85;
    margin-top: 4px;
  }

  /* Feedback badge */
  .feedback-good {
    background: linear-gradient(90deg, #003d24, #00a864);
    border: 1px solid #00e6a0;
    border-radius: 8px;
    padding: 12px 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #00e6a0;
    letter-spacing: 1px;
    text-align: center;
  }
  .feedback-warn {
    background: linear-gradient(90deg, #3d2a00, #a86a00);
    border: 1px solid #e6a000;
    border-radius: 8px;
    padding: 12px 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #e6a000;
    letter-spacing: 1px;
    text-align: center;
  }
  .feedback-bad {
    background: linear-gradient(90deg, #3d0a00, #a83000);
    border: 1px solid #e65030;
    border-radius: 8px;
    padding: 12px 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #e65030;
    letter-spacing: 1px;
    text-align: center;
  }

  /* Title */
  .app-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(90deg, #00e6a0, #00a8e6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 3px;
  }
  .app-subtitle {
    font-size: 14px;
    color: #4a6a85;
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  /* Section headers */
  .section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #00e6a0;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  /* Accuracy ring glow */
  .accuracy-high { color: #00e6a0; }
  .accuracy-mid  { color: #e6a000; }
  .accuracy-low  { color: #e65030; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid #1e3a5f;
  }

  /* Camera frame */
  .camera-container {
    border: 2px solid #1e3a5f;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(0, 200, 150, 0.1);
  }

  /* Rep counter huge display */
  .rep-display {
    font-family: 'Rajdhani', sans-serif;
    font-size: 96px;
    font-weight: 700;
    color: #00e6a0;
    text-align: center;
    line-height: 1;
    text-shadow: 0 0 40px rgba(0, 230, 160, 0.4);
  }

  /* Angle display */
  .angle-display {
    font-family: 'Rajdhani', sans-serif;
    font-size: 48px;
    font-weight: 600;
    color: #00a8e6;
    text-align: center;
    line-height: 1;
  }

  /* Stage badge */
  .stage-up {
    background: #1a2d1a;
    border: 1px solid #00e6a0;
    color: #00e6a0;
    border-radius: 20px;
    padding: 4px 16px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 2px;
    display: inline-block;
  }
  .stage-down {
    background: #1a1a2d;
    border: 1px solid #a0a0ff;
    color: #a0a0ff;
    border-radius: 20px;
    padding: 4px 16px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 2px;
    display: inline-block;
  }

  /* Buttons */
  .stButton button {
    background: linear-gradient(90deg, #00e6a0, #00a8e6) !important;
    color: #000 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
  }
  .stButton button:hover {
    box-shadow: 0 0 24px rgba(0, 230, 160, 0.5) !important;
    transform: translateY(-1px) !important;
  }

  /* Plotly dark chart containers */
  .js-plotly-plot .plotly .bg {
    fill: transparent !important;
  }

  /* Progress bars */
  .stProgress > div > div {
    background: linear-gradient(90deg, #00e6a0, #00a8e6) !important;
  }

  /* Selectbox */
  .stSelectbox select, .stSelectbox > div > div {
    background: #12192b !important;
    border-color: #1e3a5f !important;
    color: #e0e0e0 !important;
  }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "detector": None,
        "tracker": None,
        "exercise": "squat",
        "start_time": None,
        "prev_count": 0,
        "session_saved": False,
        "angle_history": [],
        "accuracy_history": [],
        "tab": "Live Trainer",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 12px">
      <div style="font-family:'Rajdhani',sans-serif;font-size:28px;font-weight:700;
                  background:linear-gradient(90deg,#00e6a0,#00a8e6);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;letter-spacing:3px">
        AI GYM TRAINER
      </div>
      <div style="color:#4a6a85;font-size:11px;letter-spacing:2px;margin-top:4px">
        POWERED BY MEDIAPIPE
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    EXERCISE_OPTIONS = {
        "🏋️ Squat": "squat",
        "💪 Push-up": "pushup",
        "🦾 Bicep Curl": "bicep_curl",
        "🦵 Lunge": "lunge",
        "🙆 Shoulder Press": "shoulder_press",
        "🏗️ Deadlift": "deadlift",
        "🛋️ Bench Press": "bench_press",
    }

    ex_label = st.selectbox("Exercise", list(EXERCISE_OPTIONS.keys()), key="ex_select")
    new_ex = EXERCISE_OPTIONS[ex_label]
    if new_ex != st.session_state.exercise:
        st.session_state.exercise = new_ex
        if st.session_state.tracker:
            st.session_state.tracker.reset()
        st.session_state.prev_count = 0
        st.session_state.angle_history = []
        st.session_state.accuracy_history = []

    st.divider()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶ START" if not st.session_state.running else "⏸ PAUSE", use_container_width=True):
            if not st.session_state.running:
                st.session_state.detector = PoseDetector()
                st.session_state.tracker = ExerciseTracker()
                st.session_state.start_time = time.time()
                st.session_state.running = True
                st.session_state.session_saved = False
                st.session_state.angle_history = []
                st.session_state.accuracy_history = []
            else:
                st.session_state.running = False

    with col_s2:
        if st.button("🔁 RESET", use_container_width=True):
            if st.session_state.tracker:
                st.session_state.tracker.reset()
            st.session_state.prev_count = 0
            st.session_state.angle_history = []
            st.session_state.accuracy_history = []

    if st.session_state.tracker and st.session_state.tracker.count > 0:
        if st.button("💾 SAVE SESSION", use_container_width=True):
            t = st.session_state.tracker
            elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
            save_workout(st.session_state.exercise, t.count, t.average_accuracy,
                         t.avg_rep_time, t.calories_burned, int(elapsed))
            st.session_state.session_saved = True
            st.success("✅ Session saved!")

    st.divider()

    # Live stats sidebar summary
    if st.session_state.tracker:
        t = st.session_state.tracker
        elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
        mins, secs = divmod(int(elapsed), 60)
        st.markdown(f"""
        <div style="color:#4a6a85;font-size:11px;letter-spacing:2px;margin-bottom:12px;
                    font-family:'Rajdhani',sans-serif;font-weight:600">LIVE SESSION</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
          <span style="color:#5a7fa0;font-size:13px">Timer</span>
          <span style="color:#e0e0e0;font-family:'Rajdhani',sans-serif;font-weight:600">
            {mins:02d}:{secs:02d}
          </span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px">
          <span style="color:#5a7fa0;font-size:13px">Avg Rep Time</span>
          <span style="color:#e0e0e0;font-family:'Rajdhani',sans-serif;font-weight:600">
            {t.avg_rep_time}s
          </span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="color:#5a7fa0;font-size:13px">Calories</span>
          <span style="color:#00e6a0;font-family:'Rajdhani',sans-serif;font-weight:600">
            {t.calories_burned} kcal
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    nav = st.radio("Navigate", ["Live Trainer", "History & Analytics"], label_visibility="collapsed")
    st.session_state.tab = nav


# ─── Main content ─────────────────────────────────────────────────────────────
if st.session_state.tab == "Live Trainer":

    # ── Header
    st.markdown("""
    <div style="padding:8px 0 20px">
      <div class="app-title">AI GYM TRAINER</div>
      <div class="app-subtitle">Computer Vision Powered Exercise Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main layout: camera | stats
    cam_col, stat_col = st.columns([3, 2], gap="large")

    with cam_col:
        st.markdown('<div class="section-header">LIVE FEED</div>', unsafe_allow_html=True)
        frame_display = st.empty()

        if not st.session_state.running:
            # Placeholder when not running
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Press START to begin", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 100, 80), 2)
            frame_display.image(placeholder, channels="BGR", use_container_width=True)

    with stat_col:
        st.markdown('<div class="section-header">LIVE STATS</div>', unsafe_allow_html=True)

        # Placeholder metrics when idle
        t = st.session_state.tracker

        rep_ph   = st.empty()
        angle_ph = st.empty()
        stage_ph = st.empty()
        fb_ph    = st.empty()

        st.markdown('<div class="section-header" style="margin-top:24px">ACCURACY</div>', unsafe_allow_html=True)
        acc_ph   = st.empty()
        acc_bar  = st.empty()

        st.markdown('<div class="section-header" style="margin-top:24px">ANGLE CHART</div>', unsafe_allow_html=True)
        chart_ph = st.empty()

    # ── Run camera loop
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        detector  = st.session_state.detector
        tracker   = st.session_state.tracker
        exercise  = st.session_state.exercise

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                frame = detector.find_pose(frame)
                lmList = detector.find_position(frame)

                if lmList:
                    tracker.update(lmList, exercise)

                    # Voice on new rep
                    if tracker.count > st.session_state.prev_count:
                        speak_rep_count(tracker.count)
                        st.session_state.prev_count = tracker.count

                    # Store history (keep last 60 frames)
                    st.session_state.angle_history.append(tracker.current_angle)
                    st.session_state.accuracy_history.append(tracker.accuracy)
                    if len(st.session_state.angle_history) > 120:
                        st.session_state.angle_history = st.session_state.angle_history[-120:]
                        st.session_state.accuracy_history = st.session_state.accuracy_history[-120:]

                    # ── Overlay on frame
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (260, 80), (10, 16, 30), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, f"REPS: {tracker.count}", (12, 36),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 230, 160), 2, cv2.LINE_AA)
                    acc_color = (0, 230, 160) if tracker.accuracy >= 80 else (30, 160, 240) if tracker.accuracy >= 50 else (60, 60, 220)
                    cv2.putText(frame, f"ACC: {tracker.accuracy}%  ANG: {int(tracker.current_angle)}", (12, 66),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, acc_color, 1, cv2.LINE_AA)

                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display.image(frame_rgb, use_container_width=True)

                # ── Stats panel update
                rep_ph.markdown(f"""
                <div style="text-align:center;padding:8px 0">
                  <div class="metric-label">REPS</div>
                  <div class="rep-display">{tracker.count}</div>
                </div>
                """, unsafe_allow_html=True)

                angle_ph.markdown(f"""
                <div style="text-align:center;padding:4px 0">
                  <div class="metric-label">JOINT ANGLE</div>
                  <div class="angle-display">{int(tracker.current_angle)}°</div>
                </div>
                """, unsafe_allow_html=True)

                stage_class = "stage-down" if tracker.stage == "DOWN" else "stage-up"
                stage_ph.markdown(f"""
                <div style="text-align:center;margin:8px 0">
                  <span class="{stage_class}">STAGE: {tracker.stage}</span>
                </div>
                """, unsafe_allow_html=True)

                fb_cls = "feedback-good" if "Good" in tracker.feedback or "Great" in tracker.feedback else \
                         "feedback-bad"  if "Too" in tracker.feedback or "ease" in tracker.feedback.lower() else \
                         "feedback-warn"
                fb_ph.markdown(f'<div class="{fb_cls}">{tracker.feedback.upper()}</div>',
                               unsafe_allow_html=True)

                acc_ph.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                  <span style="color:#5a7fa0;font-size:13px">Current Rep</span>
                  <span style="font-family:'Rajdhani',sans-serif;font-size:20px;font-weight:700;
                               color:{'#00e6a0' if tracker.accuracy>=80 else '#e6a000' if tracker.accuracy>=50 else '#e65030'}">
                    {tracker.accuracy}%
                  </span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center">
                  <span style="color:#5a7fa0;font-size:13px">Session Avg</span>
                  <span style="font-family:'Rajdhani',sans-serif;font-size:20px;font-weight:700;color:#00a8e6">
                    {tracker.average_accuracy}%
                  </span>
                </div>
                """, unsafe_allow_html=True)

                acc_bar.progress(tracker.accuracy / 100)

                # ── Angle sparkline (inline SVG — no Streamlit key conflict in loop)
                if len(st.session_state.angle_history) > 2:
                    pts = st.session_state.angle_history[-80:]
                    W, H = 400, 110
                    n = len(pts)
                    def to_xy(i, v):
                        x = int(i / max(n - 1, 1) * W)
                        y = int(H - (v / 180) * H)
                        return x, y
                    polyline = " ".join(f"{to_xy(i,v)[0]},{to_xy(i,v)[1]}" for i, v in enumerate(pts))
                    target_y = int(H - (90 / 180) * H)
                    last_val = int(pts[-1])
                    lx, ly = to_xy(n - 1, pts[-1])
                    chart_ph.markdown(f"""
                    <div style="background:rgba(12,19,30,0.8);border:1px solid #1e3a5f;
                                border-radius:10px;padding:10px 14px;">
                      <div style="font-family:'Rajdhani',sans-serif;font-size:11px;
                                  letter-spacing:2px;color:#4a6a85;margin-bottom:6px;">
                        ANGLE TRACE &nbsp;
                        <span style="color:#00e6a0;font-size:14px;font-weight:700;">{last_val}&deg;</span>
                      </div>
                      <svg viewBox="0 0 {W} {H}" width="100%" height="{H}" style="display:block;overflow:visible;">
                        <line x1="0" y1="{int(H-(45/180)*H)}"  x2="{W}" y2="{int(H-(45/180)*H)}"
                              stroke="#1e3a5f" stroke-width="0.5"/>
                        <line x1="0" y1="{int(H-(135/180)*H)}" x2="{W}" y2="{int(H-(135/180)*H)}"
                              stroke="#1e3a5f" stroke-width="0.5"/>
                        <line x1="0" y1="{target_y}" x2="{W}" y2="{target_y}"
                              stroke="#e6a000" stroke-width="1" stroke-dasharray="4 3"/>
                        <text x="{W-2}" y="{target_y - 4}" fill="#e6a000"
                              font-size="9" text-anchor="end">90 target</text>
                        <polygon points="0,{H} {polyline} {W},{H}" fill="rgba(0,230,160,0.07)"/>
                        <polyline points="{polyline}" fill="none" stroke="#00e6a0"
                                  stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>
                        <circle cx="{lx}" cy="{ly}" r="3.5" fill="#00e6a0"/>
                      </svg>
                    </div>
                    """, unsafe_allow_html=True)

                time.sleep(0.03)

        finally:
            cap.release()

# ─── History & Analytics tab ─────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="padding:8px 0 20px">
      <div class="app-title">ANALYTICS</div>
      <div class="app-subtitle">Workout History & Progress Tracking</div>
    </div>
    """, unsafe_allow_html=True)

    history = load_history()

    if not history:
        st.info("No workout history yet. Complete a session and save it to see your analytics!")
    else:
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        # ── Summary cards
        st.markdown('<div class="section-header">LIFETIME STATS</div>', unsafe_allow_html=True)
        total_reps = int(df["reps"].sum())
        total_cals = round(df["calories"].sum(), 1)
        avg_acc    = round(df[df["accuracy"] > 0]["accuracy"].mean()) if len(df) > 0 else 0
        sessions   = len(df)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, sub in [
            (c1, "TOTAL REPS",    total_reps,  "all time"),
            (c2, "CALORIES",      f"{total_cals} kcal", "burned"),
            (c3, "AVG ACCURACY",  f"{avg_acc}%", "form quality"),
            (c4, "SESSIONS",      sessions,    "completed"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts row
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="section-header">REPS OVER TIME</div>', unsafe_allow_html=True)
            daily = df.groupby(["date", "exercise"])["reps"].sum().reset_index()
            fig_reps = px.bar(
                daily, x="date", y="reps", color="exercise", barmode="stack",
                color_discrete_sequence=["#00e6a0", "#00a8e6", "#a06ee6", "#e6a000", "#e65030"],
                template="plotly_dark",
            )
            fig_reps.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,19,30,0.8)",
                legend=dict(font=dict(color="#4a6a85"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=0, b=0), height=240,
                xaxis=dict(gridcolor="rgba(30,58,95,0.5)", tickfont=dict(color="#4a6a85")),
                yaxis=dict(gridcolor="rgba(30,58,95,0.5)", tickfont=dict(color="#4a6a85")),
            )
            st.plotly_chart(fig_reps, use_container_width=True, config={"displayModeBar": False}, key="history_reps_chart")

        with right:
            st.markdown('<div class="section-header">ACCURACY TREND</div>', unsafe_allow_html=True)
            acc_df = df[df["accuracy"] > 0].copy()
            if not acc_df.empty:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=acc_df["timestamp"], y=acc_df["accuracy"],
                    mode="lines+markers",
                    line=dict(color="#00a8e6", width=2),
                    marker=dict(size=6, color="#00a8e6"),
                    fill="tozeroy", fillcolor="rgba(0,168,230,0.08)",
                ))
                fig_acc.add_hline(y=80, line_dash="dash", line_color="#00e6a0",
                                  line_width=1, annotation_text="Target 80%")
                fig_acc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,19,30,0.8)",
                    margin=dict(l=0, r=0, t=0, b=0), height=240, showlegend=False,
                    xaxis=dict(gridcolor="rgba(30,58,95,0.5)", tickfont=dict(color="#4a6a85")),
                    yaxis=dict(gridcolor="rgba(30,58,95,0.5)", tickfont=dict(color="#4a6a85"),
                               range=[0, 105]),
                )
                st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False}, key="history_acc_chart")

        # ── Per-exercise breakdown
        st.markdown('<div class="section-header" style="margin-top:24px">EXERCISE BREAKDOWN</div>',
                    unsafe_allow_html=True)
        summary = get_summary_stats(history)
        if summary:
            ex_cols = st.columns(len(summary))
            for i, (ex, stats) in enumerate(summary.items()):
                ex_name = ExerciseTracker.EXERCISE_CONFIG.get(ex, {}).get("name", ex)
                ex_cols[i].markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{ex_name}</div>
                  <div class="metric-value" style="font-size:30px">{stats['total_reps']}</div>
                  <div class="metric-sub">{stats['sessions']} sessions</div>
                  <div style="margin-top:10px;color:#00a8e6;font-family:'Rajdhani',sans-serif;font-weight:600">
                    {stats['avg_accuracy']}% accuracy
                  </div>
                  <div style="color:#4a6a85;font-size:12px">{stats['total_calories']} kcal</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Recent sessions table
        st.markdown('<div class="section-header" style="margin-top:32px">RECENT SESSIONS</div>',
                    unsafe_allow_html=True)
        show_df = df.sort_values("timestamp", ascending=False).head(10)[[
            "timestamp", "exercise", "reps", "accuracy", "avg_rep_time", "calories", "duration_s"
        ]].copy()
        show_df.columns = ["Date", "Exercise", "Reps", "Accuracy (%)", "Avg Rep (s)", "Calories", "Duration (s)"]
        show_df["Date"] = show_df["Date"].dt.strftime("%b %d, %Y %H:%M")
        st.dataframe(show_df, use_container_width=True, hide_index=True)
