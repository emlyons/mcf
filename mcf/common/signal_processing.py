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

def gaussian_2d(N: int, M: int) -> np.array:
    y = np.linspace(-2, 2, N)
    x = np.linspace(-2, 2, M)
    x, y = np.meshgrid(x, y)
    return np.exp(-((x) ** 2 + (y) ** 2) / (2 * 1 ** 2), dtype='float')

def noise(N: int, M: int, mag: float=0.01) -> np.array:
    return np.random.normal(0.0, mag, (N,M))
