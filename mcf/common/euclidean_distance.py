import numpy as np
from mcf.data_types.point import Point

""" Calculates the distance between two points using the euclidean norm (L2). """
def euclidean_distance(p1: Point, p2: Point) -> float:
    return np.sqrt((p1.y - p2.y)**2 + (p1.x - p2.x)**2)
