import numpy as np

""" Calculates the center of mass ignoring any zero elements
    Center of mass is given in y,x coordinates relative to the array. """
def get_center_of_mass(array: np.array) -> tuple[int, int]:
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
    x_center_of_mass = x_sum // x_count
    y_center_of_mass = y_sum // y_count
    return (y_center_of_mass, x_center_of_mass)
