from enum import Enum

class RegionMatchingStatus(Enum):
    SUCCESS = 0
    NO_MATCHES = 1
    EMPTY_REGION = 2
    ERROR_CONFLICTING_MATCH = 3
    ERROR_INTERNAL = 4
