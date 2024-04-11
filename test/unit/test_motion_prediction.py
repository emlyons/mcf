import unittest
from mcf.motion_prediction import motion_prediction, MotionPredictionStatus
from mcf.data_types import DetectionRegion, BoundingBox, Point

class TestMotionPrediction(unittest.TestCase):
    def setUp(self):
        self.detection = DetectionRegion(classification=1,
                                         confidence=1.0,
                                         mask=None,
                                         measured_bounding_box=BoundingBox(Point(0,0),Point(2,2)),
                                         measured_center_of_mass=Point(1,1))
        return

    def tearDown(self):
        return
    
    def predict(self):
        return motion_prediction([self.detection])

    def test_run(self):
        self.detection.velocities = [Point(10,10)]
        status = self.predict()
        self.assertEqual(MotionPredictionStatus.SUCCESS, status)

    def test_location_prediction_center_of_mass(self):
        self.detection.velocities = [Point(10,10)]
        _ = self.predict()
        next_com = self.detection.next_center_of_mass
        measure_com = self.detection.measured_center_of_mass        
        self.assertEqual(measure_com.y+10, next_com.y)
        self.assertEqual(measure_com.x+10, next_com.x)

    def test_location_prediction_bounding_box(self):
        self.detection.velocities = [Point(10,10)]
        _ = self.predict()
        next_bbox = self.detection.next_bounding_box
        measure_bbox = self.detection.measured_bounding_box        
        self.assertEqual(measure_bbox.upper_left.x + 10, next_bbox.upper_left.x)
        self.assertEqual(measure_bbox.upper_left.y + 10, next_bbox.upper_left.y)
        self.assertEqual(measure_bbox.lower_right.x + 10, next_bbox.lower_right.x)
        self.assertEqual(measure_bbox.lower_right.y + 10, next_bbox.lower_right.y)

    def test_location_prediction_center_of_mass_with_filtering(self):
        self.detection.velocities = [Point(10,10),Point(10,10),Point(10,10),Point(10,10),Point(10,10),Point(10,10)]
        _ = self.predict()
        next_com = self.detection.next_center_of_mass
        measure_com = self.detection.measured_center_of_mass        
        self.assertEqual(measure_com.y+10, next_com.y)
        self.assertEqual(measure_com.x+10, next_com.x)

    def test_location_prediction_bounding_box_with_filtering(self):
        self.detection.velocities = [Point(10,10),Point(10,10),Point(10,10),Point(10,10),Point(10,10),Point(10,10)]
        _ = self.predict()
        next_bbox = self.detection.next_bounding_box
        measure_bbox = self.detection.measured_bounding_box        
        self.assertEqual(measure_bbox.upper_left.x + 10, next_bbox.upper_left.x)
        self.assertEqual(measure_bbox.upper_left.y + 10, next_bbox.upper_left.y)
        self.assertEqual(measure_bbox.lower_right.x + 10, next_bbox.lower_right.x)
        self.assertEqual(measure_bbox.lower_right.y + 10, next_bbox.lower_right.y)

    def test_location_prediction_center_of_mass_with_filtering_and_noise(self):
        self.detection.velocities = [Point(10,21),Point(9,20),Point(11,20),Point(10,19),Point(9,19),Point(11,21)]
        _ = self.predict()
        next_com = self.detection.next_center_of_mass
        measure_com = self.detection.measured_center_of_mass        
        self.assertEqual(measure_com.y+20, next_com.y)
        self.assertEqual(measure_com.x+10, next_com.x)

    def test_location_prediction_bounding_box_with_filtering_and_noise(self):
        self.detection.velocities = [Point(10,11),Point(9,10),Point(11,10),Point(10,9),Point(9,9),Point(11,11)]
        _ = self.predict()
        next_bbox = self.detection.next_bounding_box
        measure_bbox = self.detection.measured_bounding_box        
        self.assertEqual(measure_bbox.upper_left.x + 10, next_bbox.upper_left.x)
        self.assertEqual(measure_bbox.upper_left.y + 10, next_bbox.upper_left.y)
        self.assertEqual(measure_bbox.lower_right.x + 10, next_bbox.lower_right.x)
        self.assertEqual(measure_bbox.lower_right.y + 10, next_bbox.lower_right.y)
