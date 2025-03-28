import numpy as np
from mcf.data_types.point import Point

""" Calculates the center of mass ignoring any zero elements
    Center of mass is given in y,x coordinates relative to the array. """
def get_center_of_mass(array: np.array):
    x_sum = 0
    x_count = 0
    y_sum = 0
    y_count = 0
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            if array[y,x] != 0:
                x_sum += x
                x_count += 1
                y_sum += y
                y_count += 1
    x_center_of_mass = x_sum / x_count if x_count != 0 else (array.shape[1]-1) / 2
    y_center_of_mass = y_sum / y_count if y_count != 0 else (array.shape[0]-1) / 2

    return Point(x_center_of_mass, y_center_of_mass)
