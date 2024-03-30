import unittest
from mcf.region_matching import region_matching, RegionMatchingStatus
from mcf.data_types import DetectionRegion, BoundingBox, Point

class TestMotionPrediction(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return
    
    def make_current_test_detection_region(self, com=Point(11,11), bbox=BoundingBox(Point(10,10),Point(12,12))) -> DetectionRegion:
        detection = DetectionRegion(classification=1,
                                    confidence=1.0,
                                    mask=None,
                                    bounding_box=bbox,
                                    center_of_mass=com)
        return detection
        
    def make_last_test_detection_region(self, com: Point, bbox: BoundingBox) -> DetectionRegion:
        detection = DetectionRegion(classification=1,
                                    confidence=1.0,
                                    mask=None,
                                    bounding_box=BoundingBox(Point(0,0), Point(2,2)),
                                    center_of_mass=Point(1,1),
                                    next_bounding_box=bbox,
                                    next_center_of_mass=com,
                                    velocities=[Point(2,2), Point(1,1), Point(0,0)])
        return detection

    def test_one_match(self):
        current_detection = self.make_current_test_detection_region()
        last_detection = self.make_last_test_detection_region(current_detection.center_of_mass, current_detection.bounding_box)

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], [current_detection]))
        self.assertEqual(current_detection.predicted_bounding_box, last_detection.next_bounding_box)
        self.assertEqual(current_detection.predicted_center_of_mass, last_detection.next_center_of_mass)
        self.assertEqual(current_detection.velocities, last_detection.velocities)

    def test_competing_matches_decision_by_bounding_box(self):
        current_detection = self.make_current_test_detection_region()
        last_detection_better = self.make_last_test_detection_region(current_detection.center_of_mass, current_detection.bounding_box)
        last_detection_worse = self.make_last_test_detection_region(current_detection.center_of_mass, BoundingBox(Point(10,11),Point(12,12)))
        
        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection_worse, last_detection_better], [current_detection]))
        self.assertEqual(current_detection.predicted_bounding_box, last_detection_better.next_bounding_box)
        self.assertEqual(current_detection.predicted_center_of_mass, last_detection_better.next_center_of_mass)
        self.assertEqual(current_detection.velocities, last_detection_better.velocities)

    def test_competing_matches_decision_by_center_of_mass(self):
        current_detection = self.make_current_test_detection_region()
        last_detection_better = self.make_last_test_detection_region(current_detection.center_of_mass, current_detection.bounding_box)
        last_detection_worse = self.make_last_test_detection_region(Point(10,11), current_detection.bounding_box)
        
        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection_worse, last_detection_better], [current_detection]))
        self.assertEqual(current_detection.predicted_bounding_box, last_detection_better.next_bounding_box)
        self.assertEqual(current_detection.predicted_center_of_mass, last_detection_better.next_center_of_mass)
        self.assertEqual(current_detection.velocities, last_detection_better.velocities)


    def test_two_matches(self):
        current_detection_1 = self.make_current_test_detection_region(com=Point(10,11), bbox=BoundingBox(Point(5,6), Point(15,16)))
        current_detection_2 = self.make_current_test_detection_region(com=Point(23,32), bbox=BoundingBox(Point(25,25), Point(46,43)))
        last_detection_1 = self.make_last_test_detection_region(com=Point(14,8), bbox=BoundingBox(Point(7,8), Point(18,20)))
        last_detection_2 = self.make_last_test_detection_region(com=Point(25,30), bbox=BoundingBox(Point(20,20), Point(40,45)))
        
        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection_2, last_detection_1], [current_detection_1, current_detection_2]))
        self.assertEqual(current_detection_1.predicted_bounding_box, last_detection_1.next_bounding_box)
        self.assertEqual(current_detection_1.predicted_center_of_mass, last_detection_1.next_center_of_mass)
        self.assertEqual(current_detection_1.velocities, last_detection_1.velocities)
        self.assertEqual(current_detection_2.predicted_bounding_box, last_detection_2.next_bounding_box)
        self.assertEqual(current_detection_2.predicted_center_of_mass, last_detection_2.next_center_of_mass)
        self.assertEqual(current_detection_2.velocities, last_detection_2.velocities)

    def test_unsuitable_match_distance(self):
        current_detection = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        last_detection = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(19,19), Point(21,21)))

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], [current_detection]))
        self.assertEqual(current_detection.predicted_bounding_box, None)
        self.assertEqual(current_detection.predicted_center_of_mass, None)
        self.assertEqual(current_detection.velocities, None)