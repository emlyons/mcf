import unittest
import numpy as np
from mcf.region_matching import region_matching, RegionMatchingStatus
from mcf.data_types import DetectionRegion, BoundingBox, Point
from common.test_data import gaussian_2d, noise


class TestMotionPrediction(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return
    
    def make_current_test_detection_region(self, com=Point(10,10), bbox=BoundingBox(Point(5,5),Point(15,15))) -> DetectionRegion:
        image = np.ones((100,100,3), dtype='uint8')
        mask_Y, mask_X = bbox.lower_right.y-bbox.upper_left.y, bbox.lower_right.x-bbox.upper_left.x
        detection = DetectionRegion(classification=1,
                                    confidence=1.0,
                                    mask=gaussian_2d(mask_Y, mask_X) + noise(mask_Y, mask_X),
                                    bounding_box=bbox,
                                    center_of_mass=com)
        return detection, image
        
    def make_last_test_detection_region(self, com: Point, bbox: BoundingBox) -> DetectionRegion:
        image = np.ones((100,100,3), dtype='uint8')
        mask_Y, mask_X = bbox.lower_right.y-bbox.upper_left.y, bbox.lower_right.x-bbox.upper_left.x
        detection = DetectionRegion(classification=1,
                                    confidence=1.0,
                                    mask=gaussian_2d(mask_Y, mask_X) + noise(mask_Y, mask_X),
                                    bounding_box=bbox,
                                    center_of_mass=Point(1,1),
                                    next_bounding_box=bbox,
                                    next_center_of_mass=com,
                                    velocities=[Point(2,2), Point(1,1), Point(0,0)])
        return detection, image

    def test_one_match(self):
        current_detection, current_image = self.make_current_test_detection_region()
        last_detection, last_image = self.make_last_test_detection_region(current_detection.center_of_mass, current_detection.bounding_box)

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], last_image, [current_detection], current_image))
        
        self.assertEqual(current_detection.predicted_bounding_box, last_detection.next_bounding_box)
        self.assertEqual(current_detection.predicted_center_of_mass, last_detection.next_center_of_mass)
        self.assertEqual(current_detection.velocities, last_detection.velocities)

    def test_competing_matches_decision_by_bounding_box(self):
        current_detection, current_image = self.make_current_test_detection_region()
        last_detection_better, last_image = self.make_last_test_detection_region(current_detection.center_of_mass, current_detection.bounding_box)
        last_detection_worse, _ = self.make_last_test_detection_region(current_detection.center_of_mass, BoundingBox(Point(10,11),Point(12,12)))
        
        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection_worse, last_detection_better], last_image, [current_detection], current_image))
        
        self.assertEqual(current_detection.predicted_bounding_box, last_detection_better.next_bounding_box)
        self.assertEqual(current_detection.predicted_center_of_mass, last_detection_better.next_center_of_mass)
        self.assertEqual(current_detection.velocities, last_detection_better.velocities)

    def test_competing_matches_decision_by_center_of_mass(self):
        current_detection, current_image = self.make_current_test_detection_region()
        last_detection_better, last_image = self.make_last_test_detection_region(current_detection.center_of_mass, current_detection.bounding_box)
        last_detection_worse, _ = self.make_last_test_detection_region(Point(10,11), current_detection.bounding_box)
        
        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection_worse, last_detection_better], last_image, [current_detection], current_image))
        
        self.assertEqual(current_detection.predicted_bounding_box, last_detection_better.next_bounding_box)
        self.assertEqual(current_detection.predicted_center_of_mass, last_detection_better.next_center_of_mass)
        self.assertEqual(current_detection.velocities, last_detection_better.velocities)

    def test_two_matches(self):
        current_detection_1, current_image = self.make_current_test_detection_region(com=Point(10,11), bbox=BoundingBox(Point(5,6), Point(15,16)))
        current_detection_2, _ = self.make_current_test_detection_region(com=Point(23,32), bbox=BoundingBox(Point(25,25), Point(46,43)))
        current_detections = [current_detection_1, current_detection_2]
        last_detection_1, last_image = self.make_last_test_detection_region(com=Point(14,8), bbox=BoundingBox(Point(7,8), Point(18,20)))
        last_detection_2, _ = self.make_last_test_detection_region(com=Point(25,30), bbox=BoundingBox(Point(20,20), Point(40,45)))
        last_detections = [last_detection_1, last_detection_2]
        
        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching(last_detections, last_image, current_detections, current_image))

        self.assertEqual(current_detection_1.predicted_bounding_box, last_detection_1.next_bounding_box)
        self.assertEqual(current_detection_1.predicted_center_of_mass, last_detection_1.next_center_of_mass)
        self.assertEqual(current_detection_1.velocities, last_detection_1.velocities)
        self.assertEqual(current_detection_2.predicted_bounding_box, last_detection_2.next_bounding_box)
        self.assertEqual(current_detection_2.predicted_center_of_mass, last_detection_2.next_center_of_mass)
        self.assertEqual(current_detection_2.velocities, last_detection_2.velocities)

    def test_three_matches(self):
        current_detection_1, current_image = self.make_last_test_detection_region(com=Point(5,5), bbox=BoundingBox(Point(0,0), Point(10,10)))
        current_detection_2, _ = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(15,15), Point(25,25)))
        current_detection_3, _ = self.make_last_test_detection_region(com=Point(60,40), bbox=BoundingBox(Point(55,35), Point(65,45)))
        current_detections = [current_detection_1, current_detection_2, current_detection_3]
        last_detection_1, last_image = self.make_last_test_detection_region(com=Point(5,5), bbox=BoundingBox(Point(0,0), Point(10,10)))
        last_detection_2, _ = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(15,15), Point(25,25)))
        last_detection_3, _ = self.make_last_test_detection_region(com=Point(60,40), bbox=BoundingBox(Point(55,35), Point(65,45)))
        last_detections = [last_detection_1, last_detection_2, last_detection_3]
        
        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching(last_detections, last_image, current_detections, current_image))

        self.assertEqual(current_detection_1.predicted_bounding_box, last_detection_1.next_bounding_box)
        self.assertEqual(current_detection_1.predicted_center_of_mass, last_detection_1.next_center_of_mass)
        self.assertEqual(current_detection_1.velocities, last_detection_1.velocities)
        self.assertEqual(current_detection_2.predicted_bounding_box, last_detection_2.next_bounding_box)
        self.assertEqual(current_detection_2.predicted_center_of_mass, last_detection_2.next_center_of_mass)
        self.assertEqual(current_detection_2.velocities, last_detection_2.velocities)
        self.assertEqual(current_detection_3.predicted_bounding_box, last_detection_3.next_bounding_box)
        self.assertEqual(current_detection_3.predicted_center_of_mass, last_detection_3.next_center_of_mass)
        self.assertEqual(current_detection_3.velocities, last_detection_3.velocities)

    def test_unsuitable_match_distance(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        last_detection, last_image = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(19,19), Point(21,21)))

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], last_image, [current_detection], current_image))

        self.assertEqual(current_detection.predicted_bounding_box, None)
        self.assertEqual(current_detection.predicted_center_of_mass, None)
        self.assertEqual(current_detection.velocities, None)

    def test_last_targets_but_no_current_targets(self):
        current_detections = []
        _, current_image = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        last_detection, last_image = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(19,19), Point(21,21)))

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], last_image, current_detections, current_image))

        self.assertEqual(1, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, last_detection.next_bounding_box)
        self.assertEqual(current_detections[0].predicted_center_of_mass, last_detection.next_center_of_mass)
        self.assertEqual(current_detections[0].velocities, last_detection.velocities)
        
    def test_current_targets_but_no_last_targets(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        current_detections = [current_detection]
        _, last_image = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(19,19), Point(21,21)))

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([], last_image, current_detections, current_image))

        self.assertEqual(1, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, None)
        self.assertEqual(current_detections[0].predicted_center_of_mass, None)
        self.assertEqual(current_detections[0].velocities, None)

    def test_phantom_match_and_new_target(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        current_detections = [current_detection]
        last_detection, last_image = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(19,19), Point(21,21)))

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], last_image, current_detections, current_image))

        self.assertEqual(2, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, None)
        self.assertEqual(current_detections[0].predicted_center_of_mass, None)
        self.assertEqual(current_detections[0].velocities, None)
        self.assertEqual(current_detections[1].predicted_bounding_box, last_detection.next_bounding_box)
        self.assertEqual(current_detections[1].predicted_center_of_mass, last_detection.next_center_of_mass)
        self.assertEqual(current_detections[1].velocities, last_detection.velocities)

    def test_phantom_match_and_new_target_because_it_has_left_fov(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        current_detections = [current_detection]
        last_detection, last_image = self.make_last_test_detection_region(com=Point(110,110), bbox=BoundingBox(Point(105,105), Point(115,115)))

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], last_image, current_detections, current_image))

        self.assertEqual(2, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, None)
        self.assertEqual(current_detections[0].predicted_center_of_mass, None)
        self.assertEqual(current_detections[0].velocities, None)
        self.assertEqual(current_detections[1].predicted_bounding_box, last_detection.next_bounding_box)
        self.assertEqual(current_detections[1].predicted_center_of_mass, last_detection.next_center_of_mass)
        self.assertEqual(current_detections[1].velocities, last_detection.velocities)

    def test_phantom_match_value_decays(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        current_detections = [current_detection]
        last_detection, last_image = self.make_last_test_detection_region(com=Point(110,110), bbox=BoundingBox(Point(105,105), Point(115,115)))
        last_detection.confidence = 0.0

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], last_image, current_detections, current_image))

        self.assertEqual(1, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, None)
        self.assertEqual(current_detections[0].predicted_center_of_mass, None)
        self.assertEqual(current_detections[0].velocities, None)

    def test_unmatched_because_no_predictions(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(10,10), bbox=BoundingBox(Point(9,9), Point(11,11)))
        current_detections = [current_detection]
        last_detection, last_image = self.make_last_test_detection_region(com=Point(0,0), bbox=BoundingBox(Point(0,0), Point(0,0)))
        
        # prediction values are None
        last_detection.next_center_of_mass = None
        last_detection.next_bounding_box = None

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching([last_detection], last_image, current_detections, current_image))
        
        self.assertEqual(2, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, None)
        self.assertEqual(current_detections[0].predicted_center_of_mass, None)
        self.assertEqual(current_detections[0].velocities, None)
        self.assertEqual(current_detections[1].predicted_bounding_box, last_detection.next_bounding_box)
        self.assertEqual(current_detections[1].predicted_center_of_mass, last_detection.next_center_of_mass)
        self.assertEqual(current_detections[1].velocities, last_detection.velocities)

    def test_3_last_1_current_1_match(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(23,23), bbox=BoundingBox(Point(18,18), Point(28,28)))
        current_detections = [current_detection]
        last_detection_1, last_image = self.make_last_test_detection_region(com=Point(5,5), bbox=BoundingBox(Point(0,0), Point(10,10)))
        last_detection_2, _ = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(15,15), Point(25,25)))
        last_detection_3, _ = self.make_last_test_detection_region(com=Point(60,40), bbox=BoundingBox(Point(55,35), Point(65,45)))
        last_detections = [last_detection_1, last_detection_2, last_detection_3]

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching(last_detections, last_image, current_detections, current_image))
        
        self.assertEqual(3, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, last_detection_2.next_bounding_box)
        self.assertEqual(current_detections[0].predicted_center_of_mass, last_detection_2.next_center_of_mass)
        self.assertEqual(current_detections[0].velocities, last_detection_2.velocities)
        self.assertEqual(current_detections[1].predicted_bounding_box, last_detection_1.next_bounding_box)
        self.assertEqual(current_detections[1].predicted_center_of_mass, last_detection_1.next_center_of_mass)
        self.assertEqual(current_detections[1].velocities, last_detection_1.velocities)
        self.assertEqual(current_detections[2].predicted_bounding_box, last_detection_3.next_bounding_box)
        self.assertEqual(current_detections[2].predicted_center_of_mass, last_detection_3.next_center_of_mass)
        self.assertEqual(current_detections[2].velocities, last_detection_3.velocities)

    def test_3_last_1_current_0_match(self):
        current_detection, current_image = self.make_current_test_detection_region(com=Point(15,70), bbox=BoundingBox(Point(10,65), Point(20,75)))
        current_detections = [current_detection]
        last_detection_1, last_image = self.make_last_test_detection_region(com=Point(5,5), bbox=BoundingBox(Point(0,0), Point(10,10)))
        last_detection_2, _ = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(15,15), Point(25,25)))
        last_detection_3, _ = self.make_last_test_detection_region(com=Point(60,40), bbox=BoundingBox(Point(55,35), Point(65,45)))
        last_detections = [last_detection_1, last_detection_2, last_detection_3]

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching(last_detections, last_image, current_detections, current_image))
        
        self.assertEqual(4, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, None)
        self.assertEqual(current_detections[0].predicted_center_of_mass, None)
        self.assertEqual(current_detections[0].velocities, None)
        self.assertEqual(current_detections[1].predicted_bounding_box, last_detection_1.next_bounding_box)
        self.assertEqual(current_detections[1].predicted_center_of_mass, last_detection_1.next_center_of_mass)
        self.assertEqual(current_detections[1].velocities, last_detection_1.velocities)
        self.assertEqual(current_detections[2].predicted_bounding_box, last_detection_2.next_bounding_box)
        self.assertEqual(current_detections[2].predicted_center_of_mass, last_detection_2.next_center_of_mass)
        self.assertEqual(current_detections[2].velocities, last_detection_2.velocities)
        self.assertEqual(current_detections[3].predicted_bounding_box, last_detection_3.next_bounding_box)
        self.assertEqual(current_detections[3].predicted_center_of_mass, last_detection_3.next_center_of_mass)
        self.assertEqual(current_detections[3].velocities, last_detection_3.velocities)

    def test_1_last_3_current_1_match(self):
        current_detection_1, current_image = self.make_current_test_detection_region(com=Point(5,5), bbox=BoundingBox(Point(0,0), Point(10,10)))
        current_detection_2, _ = self.make_current_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(15,15), Point(25,25)))
        current_detection_3, _ = self.make_current_test_detection_region(com=Point(60,40), bbox=BoundingBox(Point(55,35), Point(65,45)))
        current_detections = [current_detection_1, current_detection_2, current_detection_3]
        last_detection_1, last_image = self.make_last_test_detection_region(com=Point(20,20), bbox=BoundingBox(Point(15,15), Point(25,25)))
        last_detections = [last_detection_1]

        self.assertEqual(RegionMatchingStatus.SUCCESS, region_matching(last_detections, last_image, current_detections, current_image))
        
        self.assertEqual(3, len(current_detections))
        self.assertEqual(current_detections[0].predicted_bounding_box, None)
        self.assertEqual(current_detections[0].predicted_center_of_mass, None)
        self.assertEqual(current_detections[0].velocities, None)
        self.assertEqual(current_detections[1].predicted_bounding_box, last_detection_1.next_bounding_box)
        self.assertEqual(current_detections[1].predicted_center_of_mass, last_detection_1.next_center_of_mass)
        self.assertEqual(current_detections[1].velocities, last_detection_1.velocities)
        self.assertEqual(current_detections[2].predicted_bounding_box, None)
        self.assertEqual(current_detections[2].predicted_center_of_mass, None)
        self.assertEqual(current_detections[2].velocities, None)