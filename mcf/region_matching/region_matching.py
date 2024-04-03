import numpy as np
from mcf.common import MinHeap
from mcf.data_types import DetectionRegion, Match
from mcf.region_matching.region_matching_status import RegionMatchingStatus
from mcf.region_matching.criteria import get_match
from mcf.region_matching.dropout import is_valid_match, make_phantom_detection_region

def region_matching(last_detection_regions: list[DetectionRegion], last_image: np.array, current_detection_regions: list[DetectionRegion], current_image: np.array) -> RegionMatchingStatus:

    matches: list[Match] = _get_optimal_matches(last_detection_regions=last_detection_regions \
                                              , last_image=last_image \
                                              , last_available=np.arange(len(last_detection_regions)).tolist() \
                                              , current_detection_regions=current_detection_regions \
                                              , current_image=current_image \
                                              , current_available=np.arange(len(current_detection_regions)).tolist() \
                                              , memo = {})
    
    # filter invalid matches
    for idx, match in enumerate(matches):
        if not is_valid_match(match): # match threshold
            matches.pop(idx)
    status = _assign_matches(last_detection_regions=last_detection_regions, current_detection_regions=current_detection_regions, matches=matches)        

    # add phantom matches for unmatched targets from last frame
    last_unmatched = np.arange(len(last_detection_regions)).tolist()
    for match in matches:
        if match.last_index in last_unmatched:
            last_unmatched.pop(match.last_index)
    for idx in last_unmatched:
        phantom_detection = make_phantom_detection_region(last_detection_regions[idx])
        if phantom_detection is not None:
            current_detection_regions.append(phantom_detection)

    if len(current_detection_regions) == 0:
        status = RegionMatchingStatus.NO_MATCHES

    return status

def _get_optimal_matches(last_detection_regions: list[DetectionRegion], last_image: np.array, last_available: list[int], current_detection_regions: list[DetectionRegion], current_image: np.array, current_available: list[int], memo: dict[tuple]) -> RegionMatchingStatus:
    """ dynamic programming algorithm that finds the optimal (minimum cost) set of matches """
    if (len(current_available) == 0) or (len(last_available) == 0):
        return []

    match_permutation_choices = MinHeap()

    for current_index in current_available:
        current = current_detection_regions[current_index]

        for last_index in last_available:
            last = last_detection_regions[last_index]
            if is_predicted(last):

                match: Match = get_match(last=last, last_image=last_image, current=current, current_image=current_image)
                match.last_index = last_index
                match.current_index = current_index

                last_remaining = last_available.copy()
                last_remaining.remove(last_index)
                current_remaining = current_available.copy()
                current_remaining.remove(current_index)

                memo_index = _memo_hash(last_remaining, current_remaining)
                if memo_index in memo:
                    sub_match_permutation = memo[memo_index]

                else:
                    sub_match_permutation = _get_optimal_matches(last_detection_regions=last_detection_regions \
                                                            , last_image=last_image \
                                                            , last_available=last_remaining \
                                                            , current_detection_regions=current_detection_regions \
                                                            , current_image=current_image \
                                                            , current_available=current_remaining \
                                                            , memo=memo)
                    memo[memo_index] = sub_match_permutation

                if len(sub_match_permutation) > 0:
                    match.total_cost = match.cost + sub_match_permutation[0].total_cost
                    match_permutation_choices.push(match.total_cost, [match] + sub_match_permutation)
                else:
                    match.total_cost = match.cost
                    match_permutation_choices.push(match.cost, [match])
            
    if match_permutation_choices.size() > 0:
        _, best_match_permutation = match_permutation_choices.pop()
    else:
        best_match_permutation = []
    
    return best_match_permutation
    
    
def _assign_matches(last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion], matches: list[Match]) -> RegionMatchingStatus:
    for match in matches:
        last = last_detection_regions[match.last_index]
        current = current_detection_regions[match.current_index]
        
        if current.predicted_bounding_box is not None:
            return RegionMatchingStatus.ERROR_CONFLICTING_MATCH
        if current.predicted_center_of_mass is not None:
            return RegionMatchingStatus.ERROR_CONFLICTING_MATCH

        current.predicted_bounding_box = last.next_bounding_box
        current.predicted_center_of_mass = last.next_center_of_mass
        current.velocities = current.velocities + last.velocities if current.velocities else last.velocities

    return RegionMatchingStatus.SUCCESS

def _memo_hash(last_remaining: list[int], current_remaining: list[int]) -> str:
    last_vals = ''.join(map(str, last_remaining))
    current_vals = ''.join(map(str, current_remaining))
    return last_vals + '/' + current_vals

def is_predicted(detection: DetectionRegion) -> bool:
    if detection.next_bounding_box is not None and detection.next_center_of_mass is not None:
        return True
    return False

