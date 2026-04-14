import math
import numpy as np


def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3. Returns 0-360."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )
    if angle < 0:
        angle += 360
    return angle


def calculate_angle_signed(p1, p2, p3):
    """Returns angle in range [0, 180] — symmetric, joint angle style."""
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    c = np.array(p3, dtype=float)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def map_value(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another."""
    return (value - in_min) / (in_max - in_min + 1e-6) * (out_max - out_min) + out_min


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def get_midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
