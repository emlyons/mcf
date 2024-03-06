import unittest

from mcf.common.queue import Queue

class TestQueue(unittest.TestCase):
    def setUp(self):
        self.queue = Queue()

    def tearDown(self):
        return

    def test_init(self):
        self.assertIsNotNone(self.queue)

    def test_push(self):
        self.assertTrue(self.queue.size() == 0)
        
        self.queue.push(123)
        
        self.assertTrue(self.queue.size() == 1)

    def test_read(self):
        expected_value = 123
        
        self.queue.push(expected_value)
        observed_value = self.queue.front()

        self.assertEqual(expected_value, observed_value)

    def test_read_empty(self):

        self.assertTrue(self.queue.empty())

        observed_value = self.queue.front()

        self.assertEqual(None, observed_value)
        self.assertTrue(self.queue.empty())
        

    def test_pop(self):
        expected_value = 123
        
        self.queue.push(expected_value)
        observed_value = self.queue.front()

        self.queue.pop()

        self.assertTrue(self.queue.empty())



if __name__ == '__main__':
    unittest.main()
