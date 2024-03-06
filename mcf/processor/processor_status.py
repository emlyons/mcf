from enum import Enum

class ProcessorStatus(Enum):
    SUCCESS = 0
    ERROR_INTERNAL = 1
    ERROR_NOT_IMPLEMENTED = 2
    ERROR_NO_FRAMES = 3
