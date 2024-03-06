import unittest

from mcf.common.time_stamp import TimeStamp

class TestTimeStamp(unittest.TestCase):
	def setUp(self):
		return

	def tearDown(self):
		return

	def test_time_stamp_make(self):

		time_stamp: str = TimeStamp.make()
		
		self.assertTrue(len(time_stamp) > 0)

	def test_time_stamp_make_is_sequential(self):

		last_time_stamp = TimeStamp.make()
		
		for _ in range(100):
			time_stamp = TimeStamp.make()
			
			self.assertEqual(last_time_stamp, TimeStamp.get_earliest([time_stamp, last_time_stamp]))

			last_time_stamp = time_stamp


if __name__ == '__main__':
	unittest.main()
