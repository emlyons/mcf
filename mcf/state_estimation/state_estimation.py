import numpy as np
from mcf.state_estimation.state_estimation_status import StateEstimationStatus
from mcf.data_types import DetectionRegion

def state_prediction(detection_regions: list[DetectionRegion]) -> StateEstimationStatus:
    # need
    # - velocity
    # - velocty variance
    # - position
    # - position variance
    # - Filter Parameters
        # - State Transition Matrix
        # - Measurement Covariance Matrix
        # - Process Covariance Matrix
        # - Observation Matrix

    return StateEstimationStatus.SUCCESS
