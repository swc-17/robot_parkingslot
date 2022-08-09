import math
import cv2 as cv
import numpy as np
from .utils import *

def plot_points(image, pred_points):
    """Plot marking points on the image."""
    if not pred_points:
        return
    height = image.shape[0]
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        p0_x = width * marking_point.x - 0.5
        p0_y = height * marking_point.y - 0.5
        cos_val = math.cos(marking_point.direction)
        sin_val = math.sin(marking_point.direction)
        p1_x = p0_x + 50*cos_val
        p1_y = p0_y + 50*sin_val
        p2_x = p0_x - 50*sin_val
        p2_y = p0_y + 50*cos_val
        p3_x = p0_x + 50*sin_val
        p3_y = p0_y - 50*cos_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(confidence), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if marking_point.shape > 0.5:
            cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)


def plot_slots(image, pred_points, slots):
    """Plot parking slots on the image."""
    if not pred_points or not slots:
        return
    marking_points = list(list(zip(*pred_points))[1])
    height = image.shape[0]
    width = image.shape[1]
    for slot in slots:
        point_a = marking_points[slot[0]]
        point_b = marking_points[slot[1]]
        p0_x = width * point_a.x - 0.5
        p0_y = height * point_a.y - 0.5
        p1_x = width * point_b.x - 0.5
        p1_y = height * point_b.y - 0.5
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)
        distance = calc_point_squre_dist(point_a, point_b)
        if VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST:
            separating_length = LONG_SEPARATOR_LENGTH
        elif HSLOT_MIN_DIST <= distance <= HSLOT_MAX_DIST:
            separating_length = SHORT_SEPARATOR_LENGTH
        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)
