import numpy as np
import cv2 as cv
from mcf.processor.processor_status import ProcessorStatus
from mcf.detection import Detector, DetectionStatus
from mcf.motion_measurement import MotionMeasurement, MotionMeasurementStatus
from mcf.motion_prediction import motion_prediction, MotionPredictionStatus
from mcf.region_matching import region_matching, RegionMatchingStatus
from mcf.data_types import Frame
from mcf.common import Queue
from mcf.display import Display

import pickle

class Processor:

    def __init__(self, enable_display=False):
        self.queue = Queue()
        self.enable_display = enable_display
        self.display = Display()
        self.detector = Detector()
        self.motion_measurement = MotionMeasurement()
        self.count = 0

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
        
        status = self._detect(current_frame)

        if (last_frame):# proceed if last exists

            # classifier filtering w/ recursive bayes (current, last) ?? MAYBE but probably needs to be done after kalman because that is our target matching ??

            # motion measurement w/ optical flow (current, last)
            if status == ProcessorStatus.SUCCESS:
                status = self._motion_measurement(current_frame, last_frame)
        
            # motion prediction model (current, last)
            if status == ProcessorStatus.SUCCESS:
                status = self._motion_prediction(last_frame)

            if self.count == 2:
                with open('./prototype/data/matching_frames.pickle', 'wb') as f:
                    pickle.dump((current_frame, last_frame), f)
            
            # region matching, associate detections between images based on predictions to minmize error (current, last)
            if status == ProcessorStatus.SUCCESS:
                status = self._region_matching(current_frame, last_frame)

            # filtering w/ kalman or partical filter for measurement and prediction resolution (current, last)
            print('filtering')


            # motion based degredation model

            # image recovery

            # display
            print(f'count: {self.count}')
            if status == ProcessorStatus.SUCCESS and self.enable_display:
                self.display.show(current_frame, bbox=True, mask=False, velocity=True)
            
            self.count += 1
            

        return status

    def _get_frames(self) -> tuple[ProcessorStatus, Frame, Frame]:
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
    
    def _detect(self, frame: Frame) -> ProcessorStatus:
        status, detection_regions = self.detector.run(frame.image)
        if status == DetectionStatus.SUCCESS:
            frame.detection_regions = detection_regions
            status = ProcessorStatus.SUCCESS
        else:
            status = ProcessorStatus.ERROR_INTERNAL
        return status
 
    def _motion_measurement(self, frame: Frame, last_frame: Frame) -> ProcessorStatus:
        status = self.motion_measurement.run(frame.grayscale, last_frame.grayscale, frame.detection_regions)
        if status == MotionMeasurementStatus.SUCCESS:
            status = ProcessorStatus.SUCCESS
        else:
            status = ProcessorStatus.ERROR_INTERNAL
        return status
    
    def _motion_prediction(self, last_frame: Frame) -> ProcessorStatus:
        status = motion_prediction(last_frame.detection_regions)
        if status == MotionPredictionStatus.SUCCESS:
            status = ProcessorStatus.SUCCESS
        else:
            status = ProcessorStatus.ERROR_INTERNAL
        return status

    def _region_matching(self, frame: Frame, last_frame: Frame) -> ProcessorStatus:
        status = region_matching(last_detection_regions=last_frame.detection_regions \
                               , last_image=last_frame.image \
                               , current_detection_regions=frame.detection_regions \
                               , current_image=frame.image)
        if status == RegionMatchingStatus.SUCCESS:
            status = ProcessorStatus.SUCCESS
        else:
            status = ProcessorStatus.ERROR_INTERNAL
        return status
    