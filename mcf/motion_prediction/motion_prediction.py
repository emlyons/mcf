import numpy as np
from mcf.data_types import DetectionRegion, Point, BoundingBox
from mcf.motion_prediction.motion_prediction_status import MotionPredictionStatus

def motion_prediction(detection_regions: list[DetectionRegion]) -> MotionPredictionStatus:
    # predict next locations from last_detection_regions
    # update last detection regions with prediction
    # linear model: x0 + V t
    status = MotionPredictionStatus.SUCCESS
    for detection_region in detection_regions:
        velocities = detection_region.velocities
        if len(velocities) > 0:
            filter_coeffs = make_filter(min(len(velocities), 5))
            translation_vector = prediction_model(velocities, filter_coeffs)
            detection_region.next_center_of_mass = predict_center_of_mass(detection_region.measured_center_of_mass, translation_vector)
            detection_region.next_bounding_box = predict_bounding_box(detection_region.measured_bounding_box, translation_vector)
    return status

def make_filter(size) -> list[float]:
    filter_coeffs = []
    for n in range(size):
        filter_coeffs.append(np.exp(-n*0.8))
    filter_coeffs /= np.sum(filter_coeffs)
    return filter_coeffs

# returns the (y, x) translation factors from the prediction model
def prediction_model(velocities: list[Point], filter_coeffs: np.array) -> Point:
    Vy,Vx = 0,0
    
    N = min(len(velocities), len(filter_coeffs))
    for idx in range(N):
        Vx += filter_coeffs[idx] * velocities[idx].x
        Vy += filter_coeffs[idx] * velocities[idx].y
        
    Vx = int(np.round(Vx))
    Vy = int(np.round(Vy))
    velocity = Point(Vx, Vy)
    return velocity

def predict_center_of_mass(center_of_mass: Point, translation_vector: Point) -> Point:
    predicted_center_of_mass = Point(center_of_mass.x + translation_vector.x,
                                     center_of_mass.y + translation_vector.y)
    return predicted_center_of_mass

def predict_bounding_box(bounding_box: BoundingBox, translation_vector) -> BoundingBox:

    predicted_bounding_box = BoundingBox(Point(bounding_box.upper_left.x + translation_vector.x,
                                               bounding_box.upper_left.y + translation_vector.y),
                                         Point(bounding_box.lower_right.x + translation_vector.x,
                                               bounding_box.lower_right.y + translation_vector.y))
    return predicted_bounding_box
