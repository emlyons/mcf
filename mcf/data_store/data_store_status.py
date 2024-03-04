from enum import Enum

class DataStoreStatus(Enum):
    SUCCESS = 0
    ERROR_KEY_COLLISION = 2
    ERROR_INVALID_KEY = 3
