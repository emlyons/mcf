import numpy as np
from scipy.signal import correlate2d

def cross_correlate_2D(image_a: np.array, image_b: np.array) -> float:
    image_a = image_a - np.mean(image_a)
    image_b = image_b - np.mean(image_b)
    
    a_std = np.std(image_a)
    if a_std != 0:
        image_a = image_a / np.std(image_a)

    b_std = np.std(image_b)
    if b_std != 0:
        image_b = image_b / np.std(image_b)
    
    corr = correlate2d(image_a, image_b, mode='same', boundary='fill')
    corr /= np.prod(image_b.shape)
    max_corr = np.max(corr)
    return max_corr
