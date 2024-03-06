import unittest
import os
import cv2 as cv
from mcf.processor import Processor
from mcf.processor import ProcessorStatus
from mcf.frame import Frame

class TestProcessor(unittest.TestCase):
	def setUp(self):
		current_dir = os.path.dirname(os.path.abspath(__file__))
		self.image = cv.imread(current_dir + "/../data/outback.jpg")
		self.processor = Processor()

	def tearDown(self):
		return

	def test_init(self):
		self.assertIsNotNone(self.processor)

	def test_process_frame(self):

		status = self.processor.process_frame(self.image)
		
		self.assertEqual(ProcessorStatus.SUCCESS, status)

	def test_get_one_frame(self):
		expected_frame = Frame(image=self.image, grayscale=cv.cvtColor(self.image, cv.COLOR_BGR2GRAY))
		self.processor.queue.push(expected_frame)
		status, observed_frame, last_frame = self.processor._get_frames()
		
		self.assertEqual(ProcessorStatus.SUCCESS, status)
		self.assertEqual(expected_frame, observed_frame)
		self.assertEqual(None, last_frame)

	def test_get_two_frames(self):
		expected_frame_1 = Frame(image=self.image.copy(), grayscale=cv.cvtColor(self.image, cv.COLOR_BGR2GRAY))
		expected_frame_2 = Frame(image=self.image.copy(), grayscale=cv.cvtColor(self.image, cv.COLOR_BGR2GRAY))
		self.processor.queue.push(expected_frame_2)
		self.processor.queue.push(expected_frame_1)
		status, observed_frame_1, observed_frame_2 = self.processor._get_frames()
		
		self.assertEqual(ProcessorStatus.SUCCESS, status)
		self.assertEqual(expected_frame_1, observed_frame_1)
		self.assertEqual(expected_frame_2, observed_frame_2)

	def test_get_frames_empty(self):

		status, _, _ = self.processor._get_frames()
		
		self.assertEqual(ProcessorStatus.ERROR_NO_FRAMES, status)

if __name__ == '__main__':
	unittest.main()
