from mcf.data_types.bounding_box import BoundingBox

def intersection_over_union(b1: BoundingBox, b2: BoundingBox) -> float:
    intersection_area = max(0, min(b1.lower_right.x, b2.lower_right.x) \
                        - max(b1.upper_left.x, b2.upper_left.x)) * max(0, min(b1.lower_right.y, b2.lower_right.y) \
                        - max(b1.upper_left.y, b2.upper_left.y))
    union_area = b1.area() + b2.area() - intersection_area
    iou = intersection_area / union_area
    return iou
