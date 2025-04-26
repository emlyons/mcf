import numpy as np
from mcf.state_estimation.state_estimation_status import StateEstimationStatus
from mcf.data_types.detection_region import DetectionRegion
from mcf.data_types.kalman_state import KalmanState
from mcf.state_estimation.kalman_filter import run_kalman

def state_prediction(detection_regions: list[DetectionRegion]) -> StateEstimationStatus:

    for detection_region in detection_regions:
        if len(detection_region.locations) < 2 or len(detection_region.velocities) < 2:
            continue

        # get state from last tracked location
        detection_region.kalman_state.position = detection_region.locations[1]
        detection_region.kalman_state.velocity = detection_region.velocities[1]

        # average x,y velocity variance
        velocity_variance = np.array([detection_region.velocities_variance[0].x, detection_region.velocities_variance[0].y]) @ (0.5 * np.ones((2,1)))
        detection_region.kalman_state.process_covariance = velocity_variance * np.eye(4)

        # add current measurement
        detection_region.kalman_state.measured_position = detection_region.locations[0]
        detection_region.kalman_state.measured_velocity = detection_region.velocities[0]

        measurement_variance = 1 # TODO: tune parameter? consider mask region?
        detection_region.kalman_state.measurement_covariance = measurement_variance * np.eye(2)
        
        run_kalman(detection_region.kalman_state)

        detection_region.locations[0] = detection_region.kalman_state.position
    
    return StateEstimationStatus.SUCCESS

