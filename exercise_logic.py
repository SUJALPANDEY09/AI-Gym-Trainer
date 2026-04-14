from utils import calculate_angle_signed, clamp
import time


class ExerciseTracker:
    """
    Tracks reps, stage transitions, ROM, accuracy, and per-rep durations
    for 7 exercises: squat, pushup, bicep_curl, lunge, shoulder_press,
    deadlift, bench_press.

    State machine types:
      TYPE A — high angle = UP (squat, pushup, lunge, deadlift)
      TYPE B — low angle  = curled/UP (bicep_curl)
      TYPE C — high angle = pressed/UP (shoulder_press, bench_press)
    """

    EXERCISE_CONFIG = {
        "squat": {
            "landmarks": [24, 26, 28], "type": "A",
            "up_angle": 160, "down_angle": 90, "ideal_down": 90,
            "cue_too_low": 60, "cue_too_high_partial": 120, "name": "Squat",
        },
        "pushup": {
            "landmarks": [12, 14, 16], "type": "A",
            "up_angle": 155, "down_angle": 90, "ideal_down": 90,
            "cue_too_low": 50, "cue_too_high_partial": 120, "name": "Push-up",
        },
        "lunge": {
            "landmarks": [24, 26, 28], "type": "A",
            "up_angle": 160, "down_angle": 90, "ideal_down": 90,
            "cue_too_low": 60, "cue_too_high_partial": 120, "name": "Lunge",
        },
        "deadlift": {
            "landmarks": [11, 23, 25], "type": "A",
            "up_angle": 160, "down_angle": 70, "ideal_down": 60,
            "cue_too_low": 35, "cue_too_high_partial": 110, "name": "Deadlift",
        },
        "bicep_curl": {
            # LEFT arm: shoulder(11), elbow(13), wrist(15)
            # Extended arm ~160 deg (DOWN/start), curled ~40 deg (UP/top)
            "landmarks": [11, 13, 15], "type": "B",
            "up_angle": 50,    # arm fully curled = rep top
            "down_angle": 140, # arm fully extended = rep start
            "ideal_down": 40,
            "cue_too_low": 20, "cue_too_high_partial": 100, "name": "Bicep Curl",
        },
        "shoulder_press": {
            # LEFT arm: shoulder(11), elbow(13), wrist(15)
            # Elbows at shoulder height ~80 deg (DOWN), arms overhead ~160 deg (UP)
            "landmarks": [11, 13, 15], "type": "C",
            "up_angle": 155,   # arms fully pressed overhead = rep done
            "down_angle": 80,  # elbows at shoulder level = start
            "ideal_down": 75,
            "cue_too_low": 50, "cue_too_high_partial": 120, "name": "Shoulder Press",
        },
        "bench_press": {
            # RIGHT arm: shoulder(12), elbow(14), wrist(16)
            # Bar at chest ~75 deg (DOWN), arms locked ~155 deg (UP)
            "landmarks": [12, 14, 16], "type": "C",
            "up_angle": 150,   # arms locked out = rep done
            "down_angle": 80,  # bar touches chest = bottom
            "ideal_down": 75,
            "cue_too_low": 50, "cue_too_high_partial": 115, "name": "Bench Press",
        },
    }

    FEEDBACK_MESSAGES = {
        "bicep_curl": {
            "too_low": "Over-curled — control it!",
            "great":   "Full curl! Great peak!",
            "go_deeper": "Curl higher — squeeze!",
            "top":     "Extend arm fully!",
            "good":    "Good curl form!",
        },
        "shoulder_press": {
            "too_low": "Arms dropped too low!",
            "great":   "Great overhead press!",
            "go_deeper": "Press higher!",
            "top":     "Lower to shoulder height!",
            "good":    "Good press form!",
        },
        "deadlift": {
            "too_low": "Back rounding — stop!",
            "great":   "Great hinge depth!",
            "go_deeper": "Hinge more at hips!",
            "top":     "Lock out hips & glutes!",
            "good":    "Good form — drive up!",
        },
        "bench_press": {
            "too_low": "Elbows flaring — careful!",
            "great":   "Good chest touch!",
            "go_deeper": "Lower bar to chest!",
            "top":     "Lock out arms fully!",
            "good":    "Good press form!",
        },
    }

    _DEFAULT_FEEDBACK = {
        "too_low": "Too deep — ease up!",
        "great":   "Great depth!",
        "go_deeper": "Go deeper!",
        "top":     "Good stance",
        "good":    "Good form",
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.count              = 0
        self.stage              = None
        self.current_angle      = 0
        self.min_angle_this_rep = 180
        self.max_angle_this_rep = 0
        self.rep_start_time     = time.time()
        self.rep_durations      = []
        self.per_rep_accuracy   = []
        self.feedback           = "Ready"
        self.accuracy           = 100
        self.form_warnings      = []
        self._current_exercise  = "squat"

    def update(self, lmList, exercise_name):
        self._current_exercise = exercise_name
        cfg = self.EXERCISE_CONFIG.get(exercise_name)
        if cfg is None:
            return 0, 0

        ids = cfg["landmarks"]
        if any(i >= len(lmList) or lmList[i][1] == -1 for i in ids):
            self.feedback = "Stand in frame"
            return self.current_angle, self.count

        p1 = lmList[ids[0]][1:3]
        p2 = lmList[ids[1]][1:3]
        p3 = lmList[ids[2]][1:3]

        angle = calculate_angle_signed(p1, p2, p3)
        self.current_angle = angle
        self.min_angle_this_rep = min(self.min_angle_this_rep, angle)
        self.max_angle_this_rep = max(self.max_angle_this_rep, angle)

        ex_type = cfg["type"]

        if ex_type == "A":
            # High angle = standing/up. Low angle = bottom.
            if angle > cfg["up_angle"]:
                self.stage = "UP"
            if angle < cfg["down_angle"] and self.stage == "UP":
                self._complete_rep(cfg)
                self.stage = "DOWN"

        elif ex_type == "B":
            # High angle = arm extended (start). Low angle = curled (top).
            if angle > cfg["down_angle"]:
                self.stage = "DOWN"        # arm extended = ready
            if angle < cfg["up_angle"] and self.stage == "DOWN":
                self._complete_rep(cfg)
                self.stage = "UP"          # curled = rep counted

        elif ex_type == "C":
            # Low angle = start (bar at chest / elbows down). High angle = top of press.
            if angle < cfg["down_angle"]:
                self.stage = "DOWN"        # bar/arms at bottom = ready
            if angle > cfg["up_angle"] and self.stage == "DOWN":
                self._complete_rep(cfg)
                self.stage = "UP"          # fully pressed = rep counted

        self._compute_feedback(angle, cfg, exercise_name)
        return angle, self.count

    def _complete_rep(self, cfg):
        self.count += 1
        duration = time.time() - self.rep_start_time
        self.rep_durations.append(round(duration, 2))
        self.rep_start_time = time.time()

        ex_type = cfg["type"]
        if ex_type == "B":
            achieved = self.min_angle_this_rep   # deepest curl
        elif ex_type == "C":
            achieved = self.max_angle_this_rep   # highest press
        else:
            achieved = self.min_angle_this_rep   # deepest squat/pushup

        deviation = abs(achieved - cfg["ideal_down"])
        rep_acc = clamp(100 - deviation * 1.5, 0, 100)
        self.per_rep_accuracy.append(round(rep_acc))
        self.accuracy = round(rep_acc)
        self.min_angle_this_rep = 180
        self.max_angle_this_rep = 0

    def _compute_feedback(self, angle, cfg, exercise_name):
        msgs = self.FEEDBACK_MESSAGES.get(exercise_name, self._DEFAULT_FEEDBACK)
        ex_type = cfg["type"]

        if ex_type == "C":
            if angle > cfg["up_angle"]:
                self.feedback = msgs["top"]
            elif angle > cfg["cue_too_high_partial"]:
                self.feedback = msgs["good"]
            elif angle > cfg["down_angle"]:
                self.feedback = msgs["go_deeper"]
            else:
                self.feedback = msgs["good"]
        elif ex_type == "B":
            if angle < cfg["cue_too_low"]:
                self.feedback = msgs["too_low"]
            elif angle < cfg["up_angle"]:
                self.feedback = msgs["great"]
            elif angle < cfg["cue_too_high_partial"]:
                self.feedback = msgs["go_deeper"]
            elif angle > cfg["down_angle"]:
                self.feedback = msgs["top"]
            else:
                self.feedback = msgs["good"]
        else:
            if angle < cfg["cue_too_low"]:
                self.feedback = msgs["too_low"]
            elif angle < cfg["down_angle"] - 5:
                self.feedback = msgs["great"]
            elif angle < cfg["cue_too_high_partial"]:
                self.feedback = msgs["go_deeper"]
            elif angle > cfg["up_angle"]:
                self.feedback = msgs["top"]
            else:
                self.feedback = msgs["good"]

        self.form_warnings = [self.feedback]

    @property
    def average_accuracy(self):
        if not self.per_rep_accuracy:
            return 0
        return round(sum(self.per_rep_accuracy) / len(self.per_rep_accuracy))

    @property
    def avg_rep_time(self):
        if not self.rep_durations:
            return 0.0
        return round(sum(self.rep_durations) / len(self.rep_durations), 1)

    @property
    def calories_burned(self):
        weighted = {"deadlift", "bench_press", "shoulder_press"}
        rate = 0.5 if self._current_exercise in weighted else 0.3
        return round(self.count * rate, 1)
