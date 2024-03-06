from enum import Enum

class DetectionStatus(Enum):
    SUCCESS = 0
    EMPTY_FRAME = 1
    ERROR_INTERNAL = 2
