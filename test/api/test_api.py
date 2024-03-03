import unittest

from mcf import Api as mcf_api

class TestDataStore(unittest.TestCase):
	def setUp(self):
		self.api = mcf_api()

	def tearDown(self):
		return

	def test_init(self):
		self.assertIsNotNone(self.api)

if __name__ == '__main__':
	unittest.main()
