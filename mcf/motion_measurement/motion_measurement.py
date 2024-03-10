import numpy as np
import cv2 as cv
from mcf.motion_measurement.motion_measurement_status import MotionMeasurementStatus
from mcf.data_types import Frame, DetectionRegion

class MotionMeasurement:

    def __init__(self):
        self.block_size = (24,24)

        # params for feature detection 
        self.feature_params = dict(maxCorners = 50, 
                                   qualityLevel = 0.1, 
                                   blockSize = 7,
                                   minDistance = 7)
        
        # Parameters for lucas kanade optical flow 
        self.lk_params = dict(winSize = (15, 15), 
                              maxLevel = 2, 
                              criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) 


    def run(self, image_gray: Frame, image_gray_last: Frame, detection_regions: list[DetectionRegion]):
        for detection_region in detection_regions:
            mask = detection_region.mask
            bounding_box = detection_region.bounding_box
            status, end_points = self._get_features(image_gray, mask, bounding_box)

            if status == MotionMeasurementStatus.SUCCESS:
                status, motion_measurements = self._optical_flow(end_points, image_gray, image_gray_last)

            if status == MotionMeasurementStatus.SUCCESS:
                detection_region.velocity_measurement = self._refine_measurement(motion_measurements)

        return status
            
    
    def _get_features(self, gray: np.array, mask: np.array , bounding_box):
        status = MotionMeasurementStatus.SUCCESS
        (xl,yl),(xh,yh) = bounding_box
        feature_mask = np.zeros(gray.shape, dtype=np.uint8)
        feature_mask[yl:yh,xl:xh] = mask
        feature_points = cv.goodFeaturesToTrack(gray, mask=feature_mask, **self.feature_params)
        if feature_points.size == 0:
            status = MotionMeasurementStatus.NO_FEATURES

        return status, feature_points
    
    def _optical_flow(self, feature_points_current, gray_current: np.array, gray_last: np.array):
        status = MotionMeasurementStatus.SUCCESS

        feature_points_last, match_status, _ = cv.calcOpticalFlowPyrLK(gray_current, gray_last, feature_points_current, None, **self.lk_params) 

        good_current = feature_points_current[match_status == 1] # remove any unmatched features
        good_last = feature_points_last[match_status == 1]

        if good_current.size == 0:
            status = MotionMeasurementStatus.FEATURE_TRACKING_FAILED

        elif good_last.shape != good_last.shape:
            status = MotionMeasurementStatus.ERROR_INTERNAL

        motion_measurements = None
        if status == MotionMeasurementStatus.SUCCESS:
            motion_measurements = np.dstack((good_last, good_current))

        return status, motion_measurements
    
    def _refine_measurement(self, motion_points):
        motion_measurement = np.mean(motion_points[:,:,1] - motion_points[:,:,0],axis=0)
        motion_x = int(np.round(motion_measurement[0]))
        motion_y = int(np.round(motion_measurement[1]))
        return (motion_y, motion_x)
    