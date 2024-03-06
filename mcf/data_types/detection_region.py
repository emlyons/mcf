from dataclasses import dataclass

@dataclass
class DetectionRegion:
    self.bounding_box: tuple[tuple[int,int], tuple[int,int]] # upper-left coordinate, lower-right coordinate
    