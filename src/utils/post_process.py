from .utils import *
import math


def get_predicted_points(prediction):
    """Get marking points from one predicted feature map."""
    prediction = prediction.reshape((6,16,16))
    predicted_points = []

    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[0, i, j] >= POINT_THRESH:
                xval = (j + prediction[2, i, j]) / prediction.shape[2]
                yval = (i + prediction[3, i, j]) / prediction.shape[1]
                if not (BOUNDARY_THRESH <= xval <= 1-BOUNDARY_THRESH
                        and BOUNDARY_THRESH <= yval <= 1-BOUNDARY_THRESH):
                    continue
                cos_value = prediction[4, i, j]
                sin_value = prediction[5, i, j]
                direction = math.atan2(sin_value, cos_value)
                marking_point = MarkingPoint(
                    xval, yval, direction, prediction[1, i, j])
                predicted_points.append((prediction[0, i, j], marking_point))
    return non_maximum_suppression(predicted_points)

def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            if not (VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST
                    or HSLOT_MIN_DIST <= distance <= HSLOT_MAX_DIST):
                continue
            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j):
                continue
            result = pair_marking_points(point_i, point_j)
            if result == 1:
                slots.append((i, j))
            elif result == -1:
                slots.append((j, i))
    return slots




