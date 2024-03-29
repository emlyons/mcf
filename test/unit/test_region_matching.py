import unittest
from mcf.region_matching import region_matching, RegionMatchingStatus
from mcf.data_types import DetectionRegion

class TestMotionPrediction(unittest.TestCase):
    def setUp(self):
        self.detection = DetectionRegion(classification=1,
                                         confidence=1.0,
                                         bounding_box=((0,0),(2,2)),
                                         mask=None,
                                         center_of_mass=(1,1))
        return

    def tearDown(self):
        return

    # def test_run(self):
    #     self.assertEqual(RegionMatchingStatus.SUCCESS, status)
