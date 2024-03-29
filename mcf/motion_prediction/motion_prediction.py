import numpy as np
from mcf.data_types import DetectionRegion
from mcf.motion_prediction.motion_prediction_status import MotionPredictionStatus

def motion_prediction(detection_regions: list[DetectionRegion]) -> MotionPredictionStatus:
    # predict next locations from last_detection_regions
    # update last detection regions w/ prediction
    # linear model: x0 + V t
    status = MotionPredictionStatus.SUCCESS
    for detection_region in detection_regions:
        velocities = detection_region.velocities
        if velocities is not None:
            filter_coeffs = make_filter(min(len(velocities), 5))
            translation_vector = prediction_model(velocities, filter_coeffs)
            detection_region.next_center_of_mass = predict_center_of_mass(detection_region.center_of_mass, translation_vector)
            detection_region.next_bounding_box = predict_bounding_box(detection_region.bounding_box, translation_vector)
    return status

def make_filter(size):
    filter_coeffs = []
    for n in range(size):
        filter_coeffs.append(np.exp(-n*0.8))
    filter_coeffs /= np.sum(filter_coeffs)
    return filter_coeffs

# returns the (y, x) translation factors from the prediction model
def prediction_model(velocities, filter_coeffs):
    Vy,Vx = 0,0
    
    N = min(len(velocities), len(filter_coeffs))
    for idx in range(N):
        Vx += filter_coeffs[idx] * velocities[idx][0]
        Vy += filter_coeffs[idx] * velocities[idx][1]
        
    Vx = int(np.round(Vx))
    Vy = int(np.round(Vy))
    dx,dy = Vx,Vy
    return dx,dy

def predict_center_of_mass(center_of_mass, translation_vector):
    dx, dy = translation_vector
    predicted_center_of_mass = (center_of_mass[0] + dx,
                                center_of_mass[1] + dy)
    return predicted_center_of_mass

def predict_bounding_box(bounding_box, translation_vector):
    dx, dy = translation_vector
    predicted_bounding_box = ((bounding_box[0][0] + dx,
                               bounding_box[0][1] + dy),
                              (bounding_box[1][0] + dx,
                               bounding_box[1][1] + dy))
    return predicted_bounding_box
