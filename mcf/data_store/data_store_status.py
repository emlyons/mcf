from enum import Enum

class DataStoreStatus(Enum):
    SUCCESS = 0
    FIELD_COLLISION = 2
    INVALID_KEY = 3
    INVALID_FIELD = 4
