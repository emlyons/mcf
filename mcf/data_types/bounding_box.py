from mcf.data_types.point import Point
from dataclasses import dataclass

@dataclass
class BoundingBox:
    upper_left: Point
    lower_right: Point
