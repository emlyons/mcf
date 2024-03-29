import numpy as np    
from mcf.data_types.bounding_box import BoundingBox

def format_mask(mask: np.array, bounding_box: BoundingBox) -> np.array:
    y_size = bounding_box.lower_right.y - bounding_box.upper_left.y
    x_size = bounding_box.lower_right.x - bounding_box.upper_left.x
    Y = np.arange(bounding_box.upper_left.y, bounding_box.lower_right.y).reshape((-1,1)).repeat(x_size, axis=1)
    X = np.arange(bounding_box.upper_left.x, bounding_box.lower_right.x).reshape((1,-1)).repeat(y_size, axis=0)
    xy_coordinate_mask = np.dstack((Y,X))
    xy_coordinate_mask[:,:,0][mask==0] = -1
    xy_coordinate_mask[:,:,1][mask==0] = -1
    return xy_coordinate_mask
