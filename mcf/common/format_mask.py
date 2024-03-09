import numpy as np    

def format_mask(mask: np.array, bounding_box: tuple[tuple[int,int], tuple[int,int]]) -> np.array:
    ((xl, yl), (xh, yh)) = bounding_box
    y_size = yh - yl
    x_size = xh - xl
    Y = np.arange(yl, yh).reshape((-1,1)).repeat(x_size, axis=1)
    X = np.arange(xl, xh).reshape((1,-1)).repeat(y_size, axis=0)
    xy_coordinate_mask = np.dstack((Y,X))
    xy_coordinate_mask[:,:,0][mask==0] = -1
    xy_coordinate_mask[:,:,1][mask==0] = -1
    return xy_coordinate_mask
