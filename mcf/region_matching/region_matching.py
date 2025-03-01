import numpy as np
from mcf.common import MinHeap
from mcf.data_types import DetectionRegion, Match
from mcf.region_matching.region_matching_status import RegionMatchingStatus
from mcf.region_matching.criteria import get_match
from mcf.region_matching.dropout import is_valid_match, make_phantom_detection_region

def region_matching(last_detection_regions: list[DetectionRegion], last_image: np.array, current_detection_regions: list[DetectionRegion], current_image: np.array) -> RegionMatchingStatus:

    status, matches = _get_optimal_matches(last_detection_regions=last_detection_regions,
                                           last_image=last_image,
                                           last_available=np.arange(len(last_detection_regions)).tolist(),
                                           current_detection_regions=current_detection_regions,
                                           current_image=current_image,
                                           current_index=0,
                                           memo = {})
    
    if status == RegionMatchingStatus.SUCCESS:
        remove_invalid_matches(matches=matches)
        status = _assign_matches(last_detection_regions=last_detection_regions, current_detection_regions=current_detection_regions, matches=matches)        
        phantom_match_unmatched_targets(matches=matches, last_detection_regions=last_detection_regions, current_detection_regions=current_detection_regions)

        if len(current_detection_regions) == 0:
            status = RegionMatchingStatus.NO_MATCHES

    return status

def _get_optimal_matches(last_detection_regions: list[DetectionRegion], last_image: np.array, last_available: list[int], current_detection_regions: list[DetectionRegion], current_image: np.array, current_index: int, memo: dict[tuple]) -> tuple[RegionMatchingStatus, MinHeap]:
    status = RegionMatchingStatus.SUCCESS
    """ dynamic programming algorithm that finds the optimal (minimum cost) set of matches """
    if (current_index >= len(current_detection_regions)) or (len(last_available) == 0):
        return status, []

    match_permutations = MinHeap()

    if len(last_available) < (len(current_detection_regions)-current_index):
        status = skip_matching_current(last_detection_regions, last_image, last_available, current_detection_regions, current_image, current_index, memo, match_permutations)

    if status == RegionMatchingStatus.SUCCESS:
        status = match_current(last_detection_regions, last_image, last_available, current_detection_regions, current_image, current_index, memo, match_permutations)
            
    if match_permutations.size() > 0:
        _, best_match_permutation = match_permutations.pop()
    else:
        best_match_permutation = []
    
    return status, best_match_permutation

def skip_matching_current(last_detection_regions: list[DetectionRegion], last_image: np.array, last_available: list[int], current_detection_regions: list[DetectionRegion], current_image: np.array, current_index: int, memo: dict[tuple], match_permutations) -> RegionMatchingStatus:
    status = RegionMatchingStatus.SUCCESS
    memo_index = _memo_hash(last_available, [-1])
    if memo_index in memo:
        subproblem_matches = memo[memo_index]
    
    else:
        status, subproblem_matches = _get_optimal_matches(last_detection_regions=last_detection_regions,
                                                          last_image=last_image,
                                                          last_available=last_available,
                                                          current_detection_regions=current_detection_regions,
                                                          current_image=current_image,
                                                          current_index=current_index+1,
                                                          memo=memo)
        if status == RegionMatchingStatus.SUCCESS:
            memo[memo_index] = subproblem_matches
    if status == RegionMatchingStatus.SUCCESS:
        status = store_match_permutation(match_permutations, None, subproblem_matches)
    return status

def match_current(last_detection_regions: list[DetectionRegion], last_image: np.array, last_available: list[int], current_detection_regions: list[DetectionRegion], current_image: np.array, current_index: int, memo: dict[tuple], match_permutations) -> RegionMatchingStatus:
    status = RegionMatchingStatus.SUCCESS
    current = current_detection_regions[current_index]
    
    for last_index in last_available:
        if status != RegionMatchingStatus.SUCCESS:
            break

        last = last_detection_regions[last_index]
        if is_predicted(last):

            match: Match = get_match(last=last, last_image=last_image, current=current, current_image=current_image)
            match.last_index = last_index
            match.current_index = current_index

            last_remaining = last_available.copy()
            last_remaining.remove(last_index)

            memo_index = _memo_hash(last_remaining, [current_index])
            if memo_index in memo:
                subproblem_matches = memo[memo_index]

            else:
                status, subproblem_matches = _get_optimal_matches(last_detection_regions=last_detection_regions,
                                                                  last_image=last_image,
                                                                  last_available=last_remaining,
                                                                  current_detection_regions=current_detection_regions,
                                                                  current_image=current_image,
                                                                  current_index=current_index+1,
                                                                  memo=memo)
                if status == RegionMatchingStatus.SUCCESS:
                    memo[memo_index] = subproblem_matches
            if status == RegionMatchingStatus.SUCCESS:
                status = store_match_permutation(match_permutations, match, subproblem_matches)
    return status

def store_match_permutation(match_min_heap, current_match, subproblem_matches) -> RegionMatchingStatus:
    status = RegionMatchingStatus.SUCCESS
    if current_match is None and len(subproblem_matches) > 0:
        match_min_heap.push(subproblem_matches[0].total_cost, subproblem_matches)
    elif current_match is not None and len(subproblem_matches) > 0:
        current_match.total_cost = current_match.cost + subproblem_matches[0].total_cost
        match_min_heap.push(current_match.total_cost, [current_match] + subproblem_matches)    
    elif current_match is not None:
        current_match.total_cost = current_match.cost
        match_min_heap.push(current_match.total_cost, [current_match])
    else:
        status = RegionMatchingStatus.ERROR_INTERNAL
    return status
    
def _assign_matches(last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion], matches: list[Match]) -> RegionMatchingStatus:
    for match in matches:
        last: DetectionRegion = last_detection_regions[match.last_index]
        current: DetectionRegion = current_detection_regions[match.current_index]
        
        if current.matched:
            return RegionMatchingStatus.ERROR_CONFLICTING_MATCH

        current.velocities = current.velocities + last.velocities if current.velocities else last.velocities
        current.locations = current.locations + last.locations if current.locations else last.locations
        current.matched = True

    return RegionMatchingStatus.SUCCESS

def _memo_hash(last_remaining: list[int], current_remaining: list[int]) -> str:
    last_vals = ''.join(map(str, last_remaining))
    current_vals = ''.join(map(str, current_remaining))
    return last_vals + '/' + current_vals

def is_predicted(detection: DetectionRegion) -> bool:
    if detection.next_bounding_box is not None and detection.next_center_of_mass is not None:
        return True
    return False

def remove_invalid_matches(matches: list[Match]):
    N = len(matches)
    for idx in range(N-1,-1,-1):
        match = matches[idx]
        if not is_valid_match(match): # apply match threshold
            matches.pop(idx)

def phantom_match_unmatched_targets(matches: list[Match], last_detection_regions: list[DetectionRegion], current_detection_regions: list[DetectionRegion]):
    last_unmatched = np.arange(len(last_detection_regions)).tolist()
    for match in matches:
        if match.last_index in last_unmatched:
            last_unmatched.remove(match.last_index)
    for idx in last_unmatched:
        phantom_detection = make_phantom_detection_region(last_detection_regions[idx])
        if phantom_detection is not None:
            current_detection_regions.append(phantom_detection)
