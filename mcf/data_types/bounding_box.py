from mcf.data_types.point import Point
from dataclasses import dataclass

@dataclass
class BoundingBox:
    upper_left: Point
    lower_right: Point

    def area(self):
        return (self.lower_right.x - self.upper_left.x) \
             * (self.lower_right.y - self.upper_left.y)
