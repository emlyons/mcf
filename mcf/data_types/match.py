from dataclasses import dataclass

@dataclass
class Match:
    last_index: int
    current_index: int
    total_cost: float
    cost: float
    distance: float
    iou: float
    correlation: float
    