from enum import Enum

class DataStoreStatus(Enum):
    SUCCESS = 0
    ERROR_FIELD_COLLISION = 2
    ERROR_INVALID_KEY = 3
    ERROR_INVALID_FIELD = 4
