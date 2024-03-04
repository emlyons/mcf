import cv2 as cv

class Display:

    @classmethod
    def show(frame_obj):
        frame = frame_obj['frame']
        cv.imshow("", frame)
        cv.waitKey(1)
