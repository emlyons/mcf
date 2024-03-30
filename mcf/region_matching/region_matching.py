import heapq
from mcf.common import MinHeap, euclidean_distance
from mcf.data_types import DetectionRegion, BoundingBox, Point
from mcf.region_matching.region_matching_status import RegionMatchingStatus

def region_matching(last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion]) -> RegionMatchingStatus:
    status = RegionMatchingStatus.SUCCESS
    match_permutations = get_all_match_pairs(last_detection_regions, current_detection_regions)
    if match_permutations.empty():
        status = RegionMatchingStatus.EMPTY_REGION

    if status == RegionMatchingStatus.SUCCESS:
        status = assign_optimal_matches(match_permutations, last_detection_regions, current_detection_regions)

    return status

def get_all_match_pairs(last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion]) -> MinHeap:
    match_permutations = MinHeap() # min heap
    for pidx, last_detection_region in enumerate(last_detection_regions):
        for midx, current_detection_region in enumerate(current_detection_regions):
            criterion = matching_criteria(last_detection_region, current_detection_region)
            if criterion >= 0.0:
                match_permutations.push(criterion, (pidx, midx))
    return match_permutations

def assign_optimal_matches(match_permutations: MinHeap, last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion]) -> RegionMatchingStatus:
    status = RegionMatchingStatus.SUCCESS
    current_regions_to_match = {x for x in range(len(current_detection_regions))}
    last_regions_to_match = {x for x in range(len(last_detection_regions))}

    while (status == RegionMatchingStatus.SUCCESS)\
            and (len(current_regions_to_match) > 0)\
            and (len(last_regions_to_match) > 0)\
            and (match_permutations.size() > 0):

        # greedy assignment of best matches
        # TODO: is it better to look for a optimal set?
        _, (l_idx, c_idx) = match_permutations.pop()
        if (c_idx in current_regions_to_match) and (l_idx in last_regions_to_match):
            last: DetectionRegion = last_detection_regions[l_idx]
            current: DetectionRegion = current_detection_regions[c_idx]
            status = match_detection_regions(last=last, current=current)
            current_regions_to_match.remove(c_idx)
            last_regions_to_match.remove(l_idx)

    # handle dropouts in detect stage
    while len(last_regions_to_match) > 0:
        l_idx = last_regions_to_match.pop()
        last = last_detection_regions[l_idx]
        new_detection = make_phantom_detection_region(last)
        if new_detection is not None:
            current_detection_regions.append(new_detection)

    return status

def matching_criteria(detection_region_last: DetectionRegion, detection_region_current: DetectionRegion) -> float:
    predicted_com = detection_region_last.next_center_of_mass
    predicted_bbox = detection_region_last.next_bounding_box
    predicted_com = Point(predicted_com.x + predicted_bbox.upper_left.x, predicted_com.y + predicted_bbox.upper_left.y)

    measured_com = detection_region_current.center_of_mass
    measured_bbox = detection_region_current.bounding_box
    measured_com = Point(measured_com.x + measured_bbox.upper_left.x, measured_com.y + measured_bbox.upper_left.y)

    dist = euclidean_distance(predicted_com, measured_com)
    iou = intersection_over_union(predicted_bbox, measured_bbox)
    if is_valid_match():
        criterion = dist + (1-iou)
    else:
        criterion = -1
    return criterion

def match_detection_regions(last: DetectionRegion, current: DetectionRegion) -> RegionMatchingStatus:
    if current.velocities is not None:
        return RegionMatchingStatus.ERROR_CONFLICTING_MATCH
    
    if current.predicted_center_of_mass is not None:
        return RegionMatchingStatus.ERROR_CONFLICTING_MATCH
    
    if current.predicted_bounding_box is not None:
        return RegionMatchingStatus.ERROR_CONFLICTING_MATCH
    
    current.predicted_center_of_mass = last.next_center_of_mass
    current.predicted_bounding_box = last.next_bounding_box
    current.velocities = last.velocities
    return RegionMatchingStatus.SUCCESS

def intersection_over_union(b1: BoundingBox, b2: BoundingBox) -> float:
    intersection_area = max(0, min(b1.lower_right.x, b2.lower_right.x) \
                        - max(b1.upper_left.x, b2.upper_left.x)) * max(0, min(b1.lower_right.y, b2.lower_right.y) \
                        - max(b1.upper_left.y, b2.upper_left.y))
    union_area = b1.area() + b2.area() - intersection_area
    iou = intersection_area / union_area
    return iou

def is_valid_match() -> bool:
    match_status = True
    # TODO: need a way to invalidate matches
    # idea: look at variance of measurement (can estimate from cOm maybe?), and variance of prediction (velocity vector)
    #   invalid if they do no interset within a constant multiple of standard deviations
    return match_status

def make_phantom_detection_region(detection_region: DetectionRegion) -> DetectionRegion:
    if (detection_region.confidence) < 0.25: # TODO: establish threshold
        return None
    phantom_detection = DetectionRegion(classification=detection_region.classification,
                                        confidence=detection_region.confidence/2.0,
                                        mask=detection_region.mask,
                                        center_of_mass=detection_region.next_center_of_mass,
                                        bounding_box=detection_region.next_bounding_box,
                                        next_bounding_box=None,
                                        next_center_of_mass=None,
                                        predicted_bounding_box=detection_region.next_bounding_box,
                                        predicted_center_of_mass=detection_region.next_center_of_mass,
                                        velocities=detection_region.velocities)
    return phantom_detection
