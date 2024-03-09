import cv2 as cv
from mcf.data_types import Frame, DetectionRegion

GREEN = (0,255,0)
class Display:

    @classmethod
    def show(cls, frame: Frame, bbox=False):
        for detection_region in frame.detection_regions:
            top_left, bottom_right = detection_region.bounding_box
            cv.rectangle(frame.image, top_left, bottom_right, GREEN, 5)

        cv.imshow("", frame.image)
        cv.waitKey(1)
