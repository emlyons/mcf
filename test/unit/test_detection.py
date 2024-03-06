import unittest
import os
import cv2 as cv
import numpy as np
from mcf.detection import Detector, DetectionStatus

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

    def test_detection_results(self):
         return

if __name__ == '__main__':
	unittest.main()
