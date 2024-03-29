import heapq
from mcf.common import MinHeap, euclidean_distance
from mcf.data_types import DetectionRegion, BoundingBox, Point

def intersection_over_union(b1: BoundingBox, b2: BoundingBox) -> float:
    intersection_area = max(0, min(b1.lower_right.x, b2.lower_right.x) \
                        - max(b1.upper_left.x, b2.upper_left.x)) * max(0, min(b1.lower_right.y, b2.lower_right.y) \
                        - max(b1.upper_left.y, b2.upper_left.y))
    union_area = b1.area() + b2.area() - intersection_area
    iou = intersection_area / union_area
    return iou

def matching_criteria(detection_region_last: DetectionRegion, detection_region_current: DetectionRegion) -> float:
    predicted_com = detection_region_last.next_center_of_mass
    predicted_bbox = detection_region_last.next_bounding_box
    predicted_com = Point(predicted_com.x + predicted_bbox.upper_left.x, predicted_com.y + predicted_bbox.upper_left.y)

    measured_com = detection_region_current.center_of_mass
    measured_bbox = detection_region_current.bounding_box
    measured_com = Point(measured_com.x + measured_bbox.upper_left.x, measured_com.y + measured_bbox.upper_left.y)

    dist = euclidean_distance(predicted_com, measured_com)
    iou = intersection_over_union(predicted_bbox, measured_bbox)
    return dist*2*(1-iou)

def get_all_match_pairs(last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion]) -> MinHeap:
    match_permutations = MinHeap() # min heap
    for pidx, last_detection_region in enumerate(last_detection_regions):
        for midx, current_detection_region in enumerate(current_detection_regions):
            criterion = matching_criteria(last_detection_region, current_detection_region)
            match_permutations.push(criterion, (pidx, midx))
    return match_permutations

def assign_optimal_matches(match_permutations: MinHeap, last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion]):
    status = True
    regions_to_match = {x for x in range(len(current_detection_regions))}

    while (len(regions_to_match) > 0):
        _, (pidx, midx) = match_permutations.pop()

        last_dr = last_detection_regions[pidx]
        current_dr = current_detection_regions[midx]

        current_dr.last_center_of_mass = last_dr.next_center_of_mass
        current_dr.last_bounding_box = last_dr.next_bounding_box
        current_dr.velocities.append(last_dr.velocities)

        regions_to_match.remove(midx)

    return status
    
def region_matching(last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion]):
    status = True
    match_permutations = get_all_match_pairs(last_detection_regions, current_detection_regions)
    if match_permutations.empty():
        status = False

    if status:
        status = assign_optimal_matches(match_permutations, last_detection_regions, current_detection_regions)

    return status
