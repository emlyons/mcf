import unittest

import numpy as np
from mcf.common import get_center_of_mass

class TestCenterOfMass(unittest.TestCase):
	def setUp(self):
		return

	def tearDown(self):
		return

	def test_center_of_mass_of_square(self):
		square = np.ones((3,3))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, (1,1))

	def test_center_of_mass_of_square_with_zeros(self):
		square = np.zeros((5,5))
		square[1:-1,1:-1] = np.ones((3,3))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, (2,2))

	def test_center_of_mass_of_rectangle(self):
		square = np.ones((3,5))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, (1,2))

	def test_center_of_mass_of_rectangle_rounding(self):
		square = np.ones((4,8))
		center_of_mass = get_center_of_mass(square)
		self.assertEqual(center_of_mass, (1,3))


if __name__ == '__main__':
	unittest.main()
