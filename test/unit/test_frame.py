import unittest

import numpy as np
from mcf.frame import Frame

class TestFrame(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_init(self):
        image = np.random.random((3,4))
        frame = Frame(image=image, grayscale=image)
        self.assertIsNotNone(frame)

    def test_frame_with_image(self):
        image = np.random.random((3,4))
        frame = Frame(image=image.copy(), grayscale=image.copy())
        self.assertTrue(np.equal(image, frame.image).all())
        self.assertTrue(np.equal(image, frame.grayscale).all())

if __name__ == '__main__':
	unittest.main()
