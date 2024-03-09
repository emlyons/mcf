import cv2 as cv
from mcf.data_types import Frame

GREEN = (0,255,0)

def add_bounding_box(image, detection_region):
    top_left, bottom_right = detection_region.bounding_box
    cv.rectangle(image, top_left, bottom_right, GREEN, 5)

def add_mask(image, detection_region):
    top_left, bottom_right = detection_region.bounding_box
    mask = detection_region.mask
    box = image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0], 1]
    box[mask!=0] = 255
    image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0], 1] = box

class Display:

    @classmethod
    def show(cls, frame: Frame, bbox=False, mask=False):
        
        for detection_region in frame.detection_regions:
            if bbox:
                add_bounding_box(frame.image, detection_region)
            if mask:
                add_mask(frame.image, detection_region)

        cv.imshow("", frame.image)
        cv.waitKey(1)

