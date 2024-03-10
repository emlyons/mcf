import unittest
import os
import cv2 as cv
import numpy as np
from mcf.motion_measurement import MotionMeasurement, MotionMeasurementStatus
from mcf.data_types import DetectionRegion

class TestMotionMeasurement(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.image = cv.cvtColor(cv.imread(current_dir + "/../data/outback.jpg"), cv.COLOR_BGR2GRAY)
        self.blank_image = np.zeros(self.image.shape)
        self.motion_measurement = MotionMeasurement()

    def tearDown(self):
        return
    
    def shift_image(self, image: np.array, yx_shift: tuple[int, int]):
        y_shift, x_shift = yx_shift
        translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    def test_x_motion(self):
        expected_velocity = (0, 10)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)

    def test_x_motion_negative(self):
        expected_velocity = (0, -10)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)


    def test_y_motion(self):
        expected_velocity = (10, 0)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)


    def test_y_motion_negative(self):
        expected_velocity = (-10, 0)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)


    def test_xy_motion_1(self):
        expected_velocity = (10, 10)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)

    def test_xy_motion_2(self):
        expected_velocity = (-10, 10)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)

    def test_xy_motion_3(self):
        expected_velocity = (10, -10)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)


    def test_xy_motion_4(self):
        expected_velocity = (-10, -10)
        image_x_shifted = self.shift_image(self.image, expected_velocity)
        ulx, uly = 505, 330
        lrx, lry = 675, 450
        detection_region = DetectionRegion(classification=0,
                                           confidence=1.0,
                                           bounding_box=((ulx,uly),(lrx,lry)),
                                           mask=np.ones((lry-uly, lrx-ulx)),
                                           center_of_mass=(0,0))

        self.motion_measurement.run(image_gray=image_x_shifted, image_gray_last=self.image, detection_regions=[detection_region])
        observed_velocity = detection_region.velocity_measurement
        
        self.assertEqual(expected_velocity, observed_velocity)
        