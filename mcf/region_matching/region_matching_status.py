from enum import Enum

class RegionMatchingStatus(Enum):
    SUCCESS = 0
    EMPTY_REGION = 1
    ERROR_CONFLICTING_MATCH = 2
    ERROR_INTERNAL = 3
