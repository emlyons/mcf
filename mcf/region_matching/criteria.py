import numpy as np
import cv2 as cv
from mcf.data_types import DetectionRegion, BoundingBox, Point, Match
from mcf.common import euclidean_distance, trim_zero_borders, cross_correlate_2D, intersection_over_union
from mcf.region_matching.region_matching_status import RegionMatchingStatus

def get_match(last: DetectionRegion, last_image: np.array, current: DetectionRegion, current_image: np.array) -> Match:
    predicted_com = last.next_center_of_mass
    predicted_bbox = last.next_bounding_box
    predicted_com = Point(predicted_com.x + predicted_bbox.upper_left.x, predicted_com.y + predicted_bbox.upper_left.y)

    measured_com = current.measured_center_of_mass
    measured_bbox = current.measured_bounding_box
    measured_com = Point(measured_com.x + measured_bbox.upper_left.x, measured_com.y + measured_bbox.upper_left.y)

    # center of mass distance
    distance = euclidean_distance(predicted_com, measured_com)
    
    # iou of bounding box
    iou = intersection_over_union(predicted_bbox, measured_bbox)

    # mask correlation
    # correlation = correlate_mask_regions(last.mask, last.measured_bounding_box, last_image, current.mask, current.measure_bounding_box, current_image)

    match =  Match(last_index=None, last_detection=last, current_index=None, current_detection=current, total_cost=None, cost=None, distance=distance, iou=iou, correlation=1.0)
    _ = cost_function(match)
    return match

def cost_function(match: Match):
    match.cost = match.distance + match.distance*(1-match.iou) + match.distance*(1-match.correlation)
    return match.cost

def correlate_mask_regions(last_mask: np.array, last_bbox: BoundingBox, last_image: np.array, current_mask: np.array, current_bbox: BoundingBox, current_image: np.array) -> float:
    # extract patches containing masked regions from images
    status, last_mask_patch = extract_masked_patch(last_image, last_bbox, last_mask)
    if status != RegionMatchingStatus.SUCCESS:
        return 0.0

    status, current_mask_patch = extract_masked_patch(current_image, current_bbox, current_mask)
    if status != RegionMatchingStatus.SUCCESS:
        return 0.0

    # cross correlate the patches and find best match
    last_mask_patch = trim_zero_borders(last_mask_patch)
    current_mask_patch = trim_zero_borders(current_mask_patch)
    corr = cross_correlate_2D(last_mask_patch, current_mask_patch)
    return corr

def extract_masked_patch(image: np.array, bbox: BoundingBox, mask: np.array) -> np.array:
    status = RegionMatchingStatus.SUCCESS
    patch = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    upper_left = saturate_coordinates(bbox.upper_left, image.shape)
    lower_right = saturate_coordinates(bbox.lower_right, image.shape)
    if BoundingBox(upper_left, lower_right).area() > 0:
        patch = (patch[upper_left.y:lower_right.y, upper_left.x:lower_right.x]) * mask
    else:
        status = RegionMatchingStatus.EMPTY_REGION
        patch = None
    return status, patch

def saturate_coordinates(point: Point, image_shape: tuple[int, int]) -> Point:
    y = min(max(point.y, 0), image_shape[0])
    x = min(max(point.x, 0), image_shape[1])
    return Point(x, y)
