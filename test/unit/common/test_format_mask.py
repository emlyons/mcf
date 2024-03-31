import unittest

import numpy as np
from mcf.common import format_mask
from mcf.data_types import BoundingBox, Point

class TestFormatMask(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_basic_square(self):
        N = 10
        bounding_box = BoundingBox(Point(0,0),Point(N,N))
        mask = np.ones((N,N))
        formatted_mask = format_mask(mask, bounding_box)
        for y in range(0, N):
            for x in range(0, N):
                expected_yx = np.array([y,x])
                observed_yx = formatted_mask[y,x,:]
                self.assertTrue((expected_yx == observed_yx).all())

    def test_basic_square_mask_with_zeros(self):
        N = 33
        bounding_box = BoundingBox(Point(0,0),Point(N,N))
        mask = np.ones((N,N))
        for y in range(0, N):
            for x in range(0, N):
                if np.mod(y,3) == 0 and np.mod(x,2) == 0:
                    mask[y,x] = 0

        formatted_mask = format_mask(mask, bounding_box)
        for y in range(0, N):
            for x in range(0, N):
                if np.mod(y,3) == 0 and np.mod(x,2) == 0:
                    expected_yx = np.array([-1,-1])
                else:
                    expected_yx = np.array([y,x])
                observed_yx = formatted_mask[y,x,:]
                self.assertTrue((expected_yx == observed_yx).all())

    def test_basic_square_not_at_origin(self):
        N = 11
        bounding_box = BoundingBox(Point(2*N,2*N),Point(2*N+N,2*N+N))
        mask = np.ones((N,N))
        formatted_mask = format_mask(mask, bounding_box)
        for y in range(0, N):
            for x in range(0, N):
                _y = 2*N + y
                _x = 2*N + x
                expected_yx = np.array([_y,_x])
                observed_yx = formatted_mask[y,x,:]
                self.assertTrue((expected_yx == observed_yx).all())

    def test_basic_square_not_at_origin_mask_with_zeros(self):
        N = 23
        bounding_box = BoundingBox(Point(2*N,2*N),Point(2*N+N,2*N+N))
        mask = np.ones((N,N))
        for y in range(0, N):
            for x in range(0, N):
                if np.mod(y,3) == 0 and np.mod(x,2) == 0:
                    mask[y,x] = 0

        formatted_mask = format_mask(mask, bounding_box)
        for y in range(0, N):
            for x in range(0, N):
                if np.mod(y,3) == 0 and np.mod(x,2) == 0:
                    expected_yx = np.array([-1,-1])
                else:
                    _y = 2*N + y
                    _x = 2*N + x
                    expected_yx = np.array([_y,_x])
                observed_yx = formatted_mask[y,x,:]
                self.assertTrue((expected_yx == observed_yx).all())


if __name__ == '__main__':
    unittest.main()
