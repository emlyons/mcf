import numpy as np
from dataclasses import dataclass

@dataclass
class DetectionRegion:
    classification: int # most probable classifier
    confidence: float # probability of classifier
    bounding_box: tuple[tuple[int,int], tuple[int,int]] # upper-left (x,y) coordinate, lower-right (x,y) coordinate
    mask: np.ndarray # mask overlay of bounding box region
    center_of_mass: tuple[int, int] # pixel center of detected object
    velocity_measurement: tuple[int, int] = None # pixels / (seconds per frame) (y, x)
    