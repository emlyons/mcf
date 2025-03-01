import numpy as np
from mcf.data_types.point import Point
from mcf.data_types.bounding_box import BoundingBox
from dataclasses import dataclass, field

@dataclass
class DetectionRegion:
    classification: int # most probable classifier
    confidence: float # probability of classifier
    mask: np.ndarray # mask overlay of bounding box region
    measured_bounding_box: BoundingBox # measured current bounding box
    measured_center_of_mass: Point # measured current center of mass
    next_bounding_box: BoundingBox = None # predicted bounding box in next frame
    next_center_of_mass: Point = None # predicted center of mass in next frame
    predicted_bounding_box: BoundingBox = None # prediction for current bounding box
    predicted_center_of_mass: Point = None # prediction for current center of mass
    velocities: list[Point] = field(default_factory=lambda: []) # the last N velocities as, pixels / (seconds per frame)
    velocities_variance: list[Point] = field(default_factory=lambda: []) # the variance of the last N velocities
    locations: list[Point] = field(default_factory=lambda: []) # location history
