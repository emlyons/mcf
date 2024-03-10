from enum import Enum

class MotionMeasurementStatus(Enum):
    SUCCESS = 0
    NO_FEATURES = 1
    FEATURE_TRACKING_FAILED = 2
    ERROR_INTERNAL = 3
