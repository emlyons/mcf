import unittest
import numpy as np
from mcf.common import MinHeap

class TestMinHeap(unittest.TestCase):
    def setUp(self):
        self.min_heap = MinHeap()

    def tearDown(self):
        return
    
    def test_initialize(self):
        self.assertIsNotNone(self.min_heap)

    def test_empty(self):
        self.assertTrue(self.min_heap.empty())

    def test_push(self):
        self.min_heap.push(1,str(1))
        self.assertEqual(1, self.min_heap.size())

    def test_pop(self):
        self.min_heap.push(1,str(1))
        self.assertEqual((1, str(1)), self.min_heap.pop())
        self.assertTrue(self.min_heap.empty())

    def test_is_sorted(self):
        ordered = np.arange(1,10)
        unorderd = ordered.copy()
        np.random.shuffle(unorderd)
        
        for v in unorderd:
            self.min_heap.push(v, str(v))

        for v in ordered:
            self.assertEqual((v, str(v)), self.min_heap.pop())

    def test_repeated_key(self):
        N = 10
        for _ in range(N):
            self.min_heap.push(1, 'one')

        self.assertEqual(N, self.min_heap.size())
