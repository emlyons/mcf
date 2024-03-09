import numpy as np
from dataclasses import dataclass

@dataclass
class DetectionRegion:
    classification: int # most probable classifier
    confidence: float # probability of classifier
    bounding_box: tuple[tuple[int,int], tuple[int,int]] # upper-left coordinate, lower-right coordinate
    mask: np.ndarray # mask overlay of bounding box region
    center_of_mass: tuple[int, int] # pixel center of detected object
    