import cv2 as cv

class Display:

    @classmethod
    def show(cls, image):
        cv.imshow("", image)
        cv.waitKey(1)
