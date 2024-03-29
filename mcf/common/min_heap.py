import heapq

class MinHeap:

    def __init__(self):
        self._heap = []
    
    def push(self, key, val):
        heapq.heappush(self._heap, (key, val))

    def pop(self):
        if self.empty():
            return None
        else:
            return heapq.heappop(self._heap)
    
    def size(self) -> int:
        return len(self._heap)
    
    def empty(self) -> bool:
        return self.size() == 0
    