import numpy as np

def gaussian_2d(N: int, M: int) -> np.array:
    y = np.linspace(-2, 2, N)
    x = np.linspace(-2, 2, M)
    x, y = np.meshgrid(x, y)
    return np.exp(-((x) ** 2 + (y) ** 2) / (2 * 1 ** 2), dtype='float')

def noise(N: int, M: int, mag: float=0.01) -> np.array:
    return np.random.normal(0.0, mag, (N,M))