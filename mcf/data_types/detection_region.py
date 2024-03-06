import numpy as np
from dataclasses import dataclass

@dataclass
class DetectionRegion:
    classification: int # most probable classifier
    probabilities: np.array # probabilities of each classifier
    bounding_box: tuple[tuple[int,int], tuple[int,int]] # upper-left coordinate, lower-right coordinate
    mask: np.ndarray # mask overlay of bounding box region
    