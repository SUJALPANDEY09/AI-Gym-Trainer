import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class PoseDetector:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7, smooth_frames=5):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles
        self.results = None

        # Smoothing buffers per landmark
        self.smooth_frames = smooth_frames
        self.history = {}  # {lm_id: deque of (x, y)}

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mpDrawStyles.get_default_pose_landmarks_style()
            )
        return img

    def find_position(self, img, visibility_threshold=0.5):
        lmList = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                if lm.visibility < visibility_threshold:
                    lmList.append((idx, -1, -1, lm.visibility))
                    continue
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                # Smooth with rolling average
                if idx not in self.history:
                    self.history[idx] = deque(maxlen=self.smooth_frames)
                self.history[idx].append((cx, cy))
                avg_x = int(np.mean([p[0] for p in self.history[idx]]))
                avg_y = int(np.mean([p[1] for p in self.history[idx]]))

                lmList.append((idx, avg_x, avg_y, lm.visibility))
        return lmList

    def get_landmark_visibility(self, lmList, idx):
        if lmList and idx < len(lmList):
            return lmList[idx][3] if len(lmList[idx]) > 3 else 0.0
        return 0.0
