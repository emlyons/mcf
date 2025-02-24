import numpy as np
import cv2 as cv
from mcf.motion_measurement.motion_measurement_status import MotionMeasurementStatus
from mcf.data_types import Frame, DetectionRegion, BoundingBox, Point

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
        status = MotionMeasurementStatus.SUCCESS
        for detection_region in detection_regions:
            mask = detection_region.mask
            bounding_box = detection_region.measured_bounding_box
            status, feature_locations = self._get_features(image_gray, mask, bounding_box)

            if status == MotionMeasurementStatus.SUCCESS:
                status, motion_measurements = self._optical_flow(feature_locations, image_gray, image_gray_last)

            if status == MotionMeasurementStatus.SUCCESS:
                measurement, variance = self._refine_measurement(motion_measurements)
                detection_region.velocities = [measurement]
                detection_region.velocities_variance = [variance]

        return status
            
    
    def _get_features(self, gray: np.array, mask: np.array, bounding_box: BoundingBox):
        status = MotionMeasurementStatus.SUCCESS
        feature_mask = np.zeros(gray.shape, dtype=np.uint8)
        feature_mask[bounding_box.upper_left.y:bounding_box.lower_right.y, bounding_box.upper_left.x:bounding_box.lower_right.x] = mask
        feature_points = cv.goodFeaturesToTrack(gray, mask=feature_mask, **self.feature_params)
        if feature_points is None or feature_points.size == 0:
            status = MotionMeasurementStatus.NO_FEATURES
        return status, feature_points
    
    def _optical_flow(self, feature_points_current, gray_current: np.array, gray_last: np.array):
        status = MotionMeasurementStatus.SUCCESS

        feature_points_last, match_status, _ = cv.calcOpticalFlowPyrLK(gray_current, gray_last, feature_points_current, None, **self.lk_params) 

        current_points_matched = feature_points_current[match_status == 1] # remove any unmatched features
        previous_points_matched = feature_points_last[match_status == 1]

        if current_points_matched.size == 0:
            status = MotionMeasurementStatus.FEATURE_TRACKING_FAILED

        elif current_points_matched.shape != previous_points_matched.shape:
            status = MotionMeasurementStatus.ERROR_INTERNAL

        motion_measurements = None
        if status == MotionMeasurementStatus.SUCCESS:
            motion_measurements = np.dstack((previous_points_matched, current_points_matched))

        return status, motion_measurements
    
    def _refine_measurement(self, motion_points):
        motion_measurement = np.mean(motion_points[:,:,1] - motion_points[:,:,0], axis=0)
        motion_measurement_variance = np.square(np.std(motion_points[:,:,1] - motion_points[:,:,0], axis=0))
        return Point(motion_measurement[0], motion_measurement[1]), Point(motion_measurement_variance[0], motion_measurement_variance[1])
    