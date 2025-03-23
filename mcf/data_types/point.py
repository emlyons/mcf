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
    
    def __round__(self, n_digits=None):
        return Point(round(self.x, n_digits), round(self.y, n_digits))
        
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        else:
            raise TypeError("Unsupported operand type for ==: '{}' and '{}'".format(type(self), type(other)))
