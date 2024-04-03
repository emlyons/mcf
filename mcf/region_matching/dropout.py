import numpy as np
from mcf.data_types import DetectionRegion, Match

def is_valid_match(match: Match) -> bool:
    match_status = True
    if match.iou == 0:
        match_status = False
    if match.correlation < 0.5:
        match_status = False
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
