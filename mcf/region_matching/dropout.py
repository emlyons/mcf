import numpy as np
from mcf.data_types import DetectionRegion, Match

def is_valid_match(match: Match) -> bool:
    match_status = True
    
    if match.last_detection.classification != match.current_detection.classification:
        match_status = False

    if match.iou == 0:
        match_status = False

    elif match.correlation < 0.5:
        match_status = False

    return match_status

def make_phantom_detection_region(detection_region: DetectionRegion) -> DetectionRegion:
    phantom_detection = None
    if (detection_region.confidence) > 0.1: # TODO: establish threshold
        if (detection_region.next_bounding_box is not None) and (detection_region.next_center_of_mass is not None):
            phantom_detection = DetectionRegion(classification=detection_region.classification,
                                                confidence=detection_region.confidence/1.05,
                                                mask=detection_region.mask,
                                                measured_center_of_mass=detection_region.next_center_of_mass,
                                                measured_bounding_box=detection_region.next_bounding_box,
                                                next_bounding_box=None,
                                                next_center_of_mass=None,
                                                predicted_bounding_box=detection_region.next_bounding_box,
                                                predicted_center_of_mass=detection_region.next_center_of_mass,
                                                velocities=detection_region.velocities,
                                                locations=[detection_region.next_bounding_box.upper_left + detection_region.next_center_of_mass] + detection_region.locations)
    return phantom_detection
