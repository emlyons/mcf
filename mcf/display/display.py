import numpy as np
import cv2 as cv
from mcf.data_types import Frame, DetectionRegion, BoundingBox, Point

GREEN = (0,255,0)
RED = (0,0,255)
PURPLE = (200, 0, 256)

def add_bounding_box(image: np.array, detection_region: DetectionRegion):
    bbox: BoundingBox = detection_region.bounding_box
    confidence = detection_region.confidence

    # write confidence value on bounding box
    text = f'{confidence:.4f}'
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    cv.putText(image, text, (bbox.upper_left.x, bbox.upper_left.y), font, font_scale, PURPLE, font_thickness)

    # add bounding box
    cv.rectangle(image, (bbox.upper_left.x, bbox.upper_left.y), (bbox.lower_right.x, bbox.lower_right.y), GREEN, 5)

def add_mask(image: np.array, detection_region: DetectionRegion):
    bbox = detection_region.bounding_box
    mask = detection_region.mask
    box = image[bbox.upper_left.y:bbox.lower_right.y, bbox.upper_left.x:bbox.lower_right.x, 1]
    box[mask!=0] = 255
    image[bbox.upper_left.y:bbox.lower_right.y, bbox.upper_left.x:bbox.lower_right.x, 1] = box

def add_velocity(image: np.array, detection_region: DetectionRegion):
    velocity: Point = detection_region.velocities[0]
    upper_left = detection_region.bounding_box.upper_left
    start_point = (upper_left.x + detection_region.center_of_mass.x, upper_left.y + detection_region.center_of_mass.y) # xy for opencv
    end_point = start_point[0] + 20*velocity.x, start_point[1] + 20*velocity.y # scale for visual effect - xy for opencv
    image = cv.arrowedLine(image, start_point, end_point, (25,255,245), 10)

class Display:

    @classmethod
    def show(cls, frame: Frame, bbox=False, mask=False, velocity=False):
        
        for detection_region in frame.detection_regions:
            if bbox:
                add_bounding_box(frame.image, detection_region)
            if mask:
                add_mask(frame.image, detection_region)
            if velocity:
                add_velocity(frame.image, detection_region)

        cv.imshow("", frame.image)
        cv.waitKey(1)

