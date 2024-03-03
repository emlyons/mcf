import unittest

from mcf.processor.processor import Processor
from mcf.processor.processor_status import ProcessorStatus

class TestDataStore(unittest.TestCase):
	def setUp(self):
		self.processor = Processor()

	def tearDown(self):
		return

	def test_init(self):
		self.assertIsNotNone(self.processor)

	def test_configure(self):
		
		status = self.processor.configure()
		
		self.assertEqual(ProcessorStatus.ERROR_NOT_IMPLEMENTED , status)

	def test_process_frame(self):

		_ = self.processor.configure()
		status = self.processor.process_frame(None)
		
		self.assertEqual(ProcessorStatus.ERROR_NOT_IMPLEMENTED , status)

	def test_process_frame_before_configuring(self):
		
		status = self.processor.process_frame(None)
		
		self.assertEqual(ProcessorStatus.ERROR_NOT_CONFIGURED , status)

if __name__ == '__main__':
	unittest.main()
