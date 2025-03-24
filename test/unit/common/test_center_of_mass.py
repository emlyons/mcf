import unittest

import numpy as np
from mcf.common import get_center_of_mass
from mcf.data_types import Point

class TestCenterOfMass(unittest.TestCase):
	def setUp(self):
		return

	def tearDown(self):
		return

	def test_center_of_mass_of_square(self):
		square = np.ones((3,3))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, Point(1.0, 1.0))

	def test_center_of_mass_of_square_with_zeros(self):
		square = np.zeros((5,5))
		square[1:-1,1:-1] = np.ones((3,3))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, Point(2.0, 2.0))

	def test_center_of_mass_of_rectangle(self):
		square = np.ones((3,5))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, Point(2.0, 1.0))

	def test_center_of_mass_of_rectangle_rounding(self):
		square = np.ones((4,8))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, Point(3.5, 1.5))


if __name__ == '__main__':
	unittest.main()
