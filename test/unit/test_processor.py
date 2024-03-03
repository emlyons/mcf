import unittest

from mcf.processor import Processor
from mcf.processor import ProcessorStatus

class TestDataStore(unittest.TestCase):
	def setUp(self):
		self.processor = Processor()

	def tearDown(self):
		return

	def test_init(self):
		self.assertIsNotNone(self.processor)

	def test_process_frame(self):

		status = self.processor.process_frame(None)
		
		self.assertEqual(ProcessorStatus.ERROR_NOT_IMPLEMENTED , status)

if __name__ == '__main__':
	unittest.main()
