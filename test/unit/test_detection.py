import unittest
import os
import cv2 as cv
import numpy as np
from mcf.detection import Detector, DetectionStatus
from mcf.data_types import DetectionRegion, BoundingBox, Point

class TestDetection(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.image = cv.imread(current_dir + "/../data/outback.jpg")
        self.blank_image = np.zeros(self.image.shape)
        self.detector = Detector()

    def tearDown(self):
        return
    
    def run_detection(self):
        return self.detector.run(self.image)

    def test_init(self):
        self.assertIsNotNone(self.detector)

    def test_run(self):
        status, _ = self.detector.run(self.image)
        self.assertEqual(DetectionStatus.SUCCESS, status)

    def test_run_no_objects_found(self):
        status, _ = self.detector.run(self.blank_image)
        self.assertEqual(DetectionStatus.EMPTY_FRAME, status)

    def test_bounding_box_area(self):
        status, detection_regions = self.detector.run(self.image)
        self.assertEqual(status, DetectionStatus.SUCCESS)
        for detection in detection_regions:
            self.assertEqual(detection.measured_bounding_box.area(), np.prod(detection.mask.shape))

    def test_REGRESSION_detection_results(self):
        status, detection_regions = self.detector.run(self.image)
        self.assertEqual(status, DetectionStatus.SUCCESS)
        self.assertEqual(1, len(detection_regions))

        # bounding box results
        bbox = detection_regions[0].measured_bounding_box
        self.assertEqual(Point(505, 336), bbox.upper_left)
        self.assertEqual(Point(677, 442), bbox.lower_right)

        # mask results
        mask = detection_regions[0].mask
        expected_mask_size = (677-505)*(442-336) # Width x Height
        self.assertEqual(expected_mask_size, np.prod(mask.shape))
        self.assertTrue(mask[0,0] == 0)
        self.assertTrue(mask[60,60] == 1)

        # center of mass results
        center_of_mass = detection_regions[0].measured_center_of_mass
        self.assertEqual(Point(86, 49), center_of_mass)
        

if __name__ == '__main__':
	unittest.main()
