from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def as_tuple(self):
        return (self.x, self.y)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Unsupported operand type for +: '{}' and '{}'".format(type(self), type(other)))
