import unittest
from mcf.motion_prediction import motion_prediction, MotionPredictionStatus
from mcf.data_types import DetectionRegion, BoundingBox, Point

class TestMotionPrediction(unittest.TestCase):
    def setUp(self):
        self.detection = DetectionRegion(classification=1,
                                         confidence=1.0,
                                         mask=None,
                                         bounding_box=BoundingBox(Point(0,0),Point(2,2)),
                                         center_of_mass=Point(1,1))
        return

    def tearDown(self):
        return
    
    # def predict(self):
    #     return motion_prediction([self.detection])

    def test_run(self):
        self.detection.velocities = [(10,10)]
        # status = self.predict()
        # self.assertEqual(MotionPredictionStatus.SUCCESS, status)

    # def test_location_prediction_center_of_mass(self):
    #     self.detection.velocities = [(10,10)]
    #     _ = self.predict()
    #     observed_y, obversed_x = self.detection.next_center_of_mass
    #     last_y, last_x = self.detection.center_of_mass
    #     expected_y, expected_x = last_y+10, last_x+10
        
    #     self.assertEqual(expected_y, observed_y)
    #     self.assertEqual(expected_x, obversed_x)

    # def test_location_prediction_bounding_box(self):
    #     self.detection.velocities = [(10,10)]
    #     _ = self.predict()
    #     (x0,y0),(x1,y1) = self.detection.next_bounding_box
    #     (lx0,ly0),(lx1,ly1) = self.detection.bounding_box
    #     expected_x0 = lx0 + 10
    #     expected_y0 = ly0 + 10
    #     expected_x1 = lx1 + 10
    #     expected_y1 = ly1 + 10
        
    #     self.assertEqual(expected_x0, x0)
    #     self.assertEqual(expected_y0, y0)
    #     self.assertEqual(expected_x1, x1)
    #     self.assertEqual(expected_y1, y1)

    # def test_location_prediction_center_of_mass_with_filtering(self):
    #     self.detection.velocities = [(10,10),(10,10),(10,10),(10,10),(10,10),(10,10)]
    #     _ = self.predict()
    #     observed_y, obversed_x = self.detection.next_center_of_mass
    #     last_y, last_x = self.detection.center_of_mass
    #     expected_y, expected_x = last_y+10, last_x+10
        
    #     self.assertEqual(expected_y, observed_y)
    #     self.assertEqual(expected_x, obversed_x)

    # def test_location_prediction_bounding_box_with_filtering(self):
    #     self.detection.velocities = [(10,10),(10,10),(10,10),(10,10),(10,10),(10,10)]
    #     _ = self.predict()
    #     (x0,y0),(x1,y1) = self.detection.next_bounding_box
    #     (lx0,ly0),(lx1,ly1) = self.detection.bounding_box
    #     expected_x0 = lx0 + 10
    #     expected_y0 = ly0 + 10
    #     expected_x1 = lx1 + 10
    #     expected_y1 = ly1 + 10
        
    #     self.assertEqual(expected_x0, x0)
    #     self.assertEqual(expected_y0, y0)
    #     self.assertEqual(expected_x1, x1)
    #     self.assertEqual(expected_y1, y1)

    # def test_location_prediction_center_of_mass_with_filtering_and_noise(self):
    #     self.detection.velocities = [(10,11),(9,10),(11,10),(10,9),(9,9),(11,11)]
    #     _ = self.predict()
    #     observed_y, obversed_x = self.detection.next_center_of_mass
    #     last_y, last_x = self.detection.center_of_mass
    #     expected_y, expected_x = last_y+10, last_x+10
        
    #     self.assertEqual(expected_y, observed_y)
    #     self.assertEqual(expected_x, obversed_x)

    # def test_location_prediction_bounding_box_with_filtering_and_noise(self):
    #     self.detection.velocities = [(10,11),(9,10),(11,10),(10,9),(9,9),(11,11)]
    #     _ = self.predict()
    #     (x0,y0),(x1,y1) = self.detection.next_bounding_box
    #     (lx0,ly0),(lx1,ly1) = self.detection.bounding_box
    #     expected_x0 = lx0 + 10
    #     expected_y0 = ly0 + 10
    #     expected_x1 = lx1 + 10
    #     expected_y1 = ly1 + 10
        
    #     self.assertEqual(expected_x0, x0)
    #     self.assertEqual(expected_y0, y0)
    #     self.assertEqual(expected_x1, x1)
    #     self.assertEqual(expected_y1, y1)
