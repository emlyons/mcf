import numpy as np
import cv2 as cv
from mcf.data_types import Frame, DetectionRegion, BoundingBox, Point

GREEN = (0,255,0)
RED = (0,0,255)
PURPLE = (200, 0, 256)
YELLOW = (25,255,245)

def add_bounding_box(image: np.array, detection_region: DetectionRegion, bbox_color=GREEN, text_color=PURPLE):
    bbox: BoundingBox = detection_region.measured_bounding_box
    confidence = detection_region.confidence

    # write confidence value on bounding box
    text = f'{confidence:.4f}'
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    cv.putText(image, text, (bbox.upper_left.x, bbox.upper_left.y), font, font_scale, text_color, font_thickness)

    # add bounding box
    cv.rectangle(image, (bbox.upper_left.x, bbox.upper_left.y), (bbox.lower_right.x, bbox.lower_right.y), bbox_color, 5)

def add_mask(image: np.array, detection_region: DetectionRegion):
    bbox = detection_region.measured_bounding_box
    mask = detection_region.mask
    box = image[bbox.upper_left.y:bbox.lower_right.y, bbox.upper_left.x:bbox.lower_right.x, 1]
    box[mask!=0] = 255
    image[bbox.upper_left.y:bbox.lower_right.y, bbox.upper_left.x:bbox.lower_right.x, 1] = box

def add_velocity(image: np.array, detection_region: DetectionRegion):
    if (len(detection_region.velocities) > 0):
        velocity: Point = detection_region.velocities[0]
        upper_left = detection_region.measured_bounding_box.upper_left
        start_point = int(upper_left.x + detection_region.measured_center_of_mass.x), int(upper_left.y + detection_region.measured_center_of_mass.y) # xy for opencv
        end_point = int(start_point[0] + 20*velocity.x), int(start_point[1] + 20*velocity.y) # scale for visual effect - xy for opencv
        image = cv.arrowedLine(image, start_point, end_point, YELLOW, 10)

def add_location_history(image: np.array, detection_region: DetectionRegion):
    if detection_region.locations is not None and len(detection_region.locations) > 1:
        for idx in range(len(detection_region.locations)-1):
            start_point = detection_region.locations[idx].x, detection_region.locations[idx].y
            end_point = detection_region.locations[idx+1].x, detection_region.locations[idx+1].y
            p1 = tuple([int(start_point[0]), int(start_point[1])])
            p2 = tuple([int(end_point[0]), int(end_point[1])])
            image = cv.arrowedLine(image, p1, p2, RED, 10)

class Display:

    @classmethod
    def show(cls, frame: Frame, bbox=False, mask=False, velocity=False):
        display_image = frame.image.copy()
        for detection_region in frame.detection_regions:
            if bbox:
                add_bounding_box(display_image, detection_region)
            if mask:
                add_mask(display_image, detection_region)
            if velocity:
                add_velocity(display_image, detection_region)

            add_location_history(display_image, detection_region)

        cv.imshow("", display_image)
        cv.waitKey(1)

