import numpy as np
import cv2 as cv
from mcf.processor.processor_status import ProcessorStatus
from mcf.detection import Detector
from mcf.frame import Frame
from mcf.common.queue import Queue
from mcf.display import Display

class Processor:

    def __init__(self, enable_display=False):
        self.queue = Queue()
        self.enable_display = enable_display
        self.display = Display()
        self.detector = Detector()

    def process_frame(self, image: np.array) -> ProcessorStatus:
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        frame = Frame(image=image, grayscale=grayscale)
        self.queue.push(frame)
        status = self._pipeline()
        return status
    
    def _pipeline(self) -> ProcessorStatus:
        status = ProcessorStatus.SUCCESS

        # get current and last -> remove last if it exists
        status, current_frame, last_frame = self._get_frames()
        
        self._detect(current_frame)

        if (last_frame):# proceed if last exists

            # classifier filtering w/ recursive bayes (current, last)

            # motion measurement w/ optical flow (current, last)
        
            # motion prediction model (current, last)
            
            # region matching, associate detections between images based on predictions to minmize error (current, last)

            # filtering w/ kalman or partical filter for measurement and prediction resolution (current, last)

            # motion based degredation model

            # image recovery

            # display
            if self.enable_display:
                self.display.show(current_frame.image)
                
        return status

    def _get_frames(self):
        status = ProcessorStatus.SUCCESS
        last_frame = None
        current_frame = None
        if self.queue.size() > 0:
            if self.queue.size() > 1:
                last_frame = self.queue.front()
                self.queue.pop()
            current_frame = self.queue.front()
        else:
            status = ProcessorStatus.ERROR_NO_FRAMES
        return status, current_frame, last_frame
    
    def _detect(self, frame):
        self.detector.run(frame.image)
        # TODO: parse results to frame object
        # need detection region
        # perhaps mask truncated by bounding box
        # center of mass could be useful to calculate here
        # also store probability vector for classifier filtering
