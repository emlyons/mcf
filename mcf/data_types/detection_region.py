import numpy as np
from dataclasses import dataclass

@dataclass
class DetectionRegion:
    classification: int # most probable classifier
    confidence: float # probability of classifier
    bounding_box: tuple[tuple[int,int], tuple[int,int]] # upper-left (x,y) coordinate, lower-right (x,y) coordinate
    mask: np.ndarray # mask overlay of bounding box region
    center_of_mass: tuple[int, int] # pixel center of detected object (y, x)
    velocities: list[tuple[int, int]] = None # the last N velocities as, pixels / (seconds per frame) (y, x), index0 is current velocity mesaure for this frame
    next_bounding_box: tuple[tuple[int,int], tuple[int,int]] = None # predicted next upper-left (x,y) coordinate, lower-right (x,y) coordinate
    next_center_of_mass: tuple[int, int] = None # predicted next pixel center of detected object (y, x)
    