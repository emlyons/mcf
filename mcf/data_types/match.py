from dataclasses import dataclass
from mcf.data_types.detection_region import DetectionRegion

@dataclass
class Match:
    last_index: int
    last_detection: DetectionRegion
    current_index: int
    current_detection: DetectionRegion
    total_cost: float
    cost: float
    distance: float
    iou: float
    correlation: float
    