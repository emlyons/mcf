import cv2 as cv
from mcf.data_types import Frame, DetectionRegion

GREEN = (0,255,0)
RED = (0,0,255)
PURPLE = (200, 0, 256)

def add_bounding_box(image, detection_region: DetectionRegion):
    top_left, bottom_right = detection_region.bounding_box
    confidence = detection_region.confidence

    # write confidence on image
    text = f'{confidence:.4f}'
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    cv.putText(image, text, (top_left[0], top_left[1]), font, font_scale, PURPLE, font_thickness)

    # add bounding box
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

