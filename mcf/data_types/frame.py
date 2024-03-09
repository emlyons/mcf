import numpy as np
from dataclasses import dataclass
from mcf.common import TimeStamp
from mcf.data_types.detection_region import DetectionRegion

@dataclass
class Frame:
    image: np.array
    grayscale: np.array
    detection_regions: list[DetectionRegion] = None
    timestamp: str = TimeStamp.make()
