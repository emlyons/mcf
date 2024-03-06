import numpy as np
from dataclasses import dataclass
from mcf.common.time_stamp import TimeStamp

@dataclass
class Frame:
    image: np.array
    grayscale: np.array
    timestamp: str = TimeStamp.make()
