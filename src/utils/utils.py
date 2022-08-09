from collections import namedtuple
from enum import Enum
import numpy as np
import math


BOUNDARY_THRESH = 0.05
POINT_THRESH = 0.11676871

BOUNDARY_THRESH = 0.01
POINT_THRESH = 0.01

VSLOT_MIN_DIST = 0.044771278151623496
VSLOT_MAX_DIST = 0.1099427457599304
HSLOT_MIN_DIST = 0.15057789144568634
HSLOT_MAX_DIST = 0.44449496544202816

SHORT_SEPARATOR_LENGTH = 0.199519231
LONG_SEPARATOR_LENGTH = 0.46875

SLOT_SUPPRESSION_DOT_PRODUCT_THRESH = 0.8
BRIDGE_ANGLE_DIFF = 0.09757113548987695 + 0.1384059287593468
SEPARATOR_ANGLE_DIFF = 0.284967562063968 + 0.1384059287593468

MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction', 'shape'])
Slot = namedtuple('Slot', ['x1', 'y1', 'x2', 'y2'])


class PointShape(Enum):
    """The point shape types used to pair two marking points into slot."""
    none = 0
    l_down = 1
    t_down = 2
    t_middle = 3
    t_up = 4
    l_up = 5


def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            i_x = pred_points[i][1].x
            i_y = pred_points[i][1].y
            j_x = pred_points[j][1].x
            j_y = pred_points[j][1].y
            # 0.0625 = 1 / 16
            if abs(j_x - i_x) < 0.0625 and abs(j_y - i_y) < 0.0625:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points


def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2*math.pi - diff


def detemine_point_shape(point, vector):
    """Determine which category the point is in."""
    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    if point.shape < 0.5:
        if direction_diff(vec_direct, point.direction) < BRIDGE_ANGLE_DIFF:
            return PointShape.t_middle
        if direction_diff(vec_direct_up, point.direction) < SEPARATOR_ANGLE_DIFF:
            return PointShape.t_up
        if direction_diff(vec_direct_down, point.direction) < SEPARATOR_ANGLE_DIFF:
            return PointShape.t_down
    else:
        if direction_diff(vec_direct, point.direction) < BRIDGE_ANGLE_DIFF:
            return PointShape.l_down
        if direction_diff(vec_direct_up, point.direction) < SEPARATOR_ANGLE_DIFF:
            return PointShape.l_up
    return PointShape.none


def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    return distx ** 2 + disty ** 2


def pass_through_third_point(marking_points, i, j):
    """See whether the line between two points pass through a third point."""
    x_1 = marking_points[i].x
    y_1 = marking_points[i].y
    x_2 = marking_points[j].x
    y_2 = marking_points[j].y
    for point_idx, point in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        x_0 = point.x
        y_0 = point.y
        vec1 = np.array([x_0 - x_1, y_0 - y_1])
        vec2 = np.array([x_2 - x_0, y_2 - y_0])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        if np.dot(vec1, vec2) > SLOT_SUPPRESSION_DOT_PRODUCT_THRESH:
            return True
    return False

def pair_marking_points(point_a, point_b):
    """See whether two marking points form a slot."""
    vector_ab = np.array([point_b.x - point_a.x, point_b.y - point_a.y])
    vector_ab = vector_ab / np.linalg.norm(vector_ab)
    point_shape_a = detemine_point_shape(point_a, vector_ab)
    point_shape_b = detemine_point_shape(point_b, -vector_ab)
    if point_shape_a.value == 0 or point_shape_b.value == 0:
        return 0
    if point_shape_a.value == 3 and point_shape_b.value == 3:
        return 0
    if point_shape_a.value > 3 and point_shape_b.value > 3:
        return 0
    if point_shape_a.value < 3 and point_shape_b.value < 3:
        return 0
    if point_shape_a.value != 3:
        if point_shape_a.value > 3:
            return 1
        if point_shape_a.value < 3:
            return -1
    if point_shape_a.value == 3:
        if point_shape_b.value < 3:
            return 1
        if point_shape_b.value > 3:
            return -1


