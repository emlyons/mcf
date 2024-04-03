import heapq

class MinHeap:

    def __init__(self):
        self._heap = []
    
    def push(self, key, val):
        if not self.empty():
            for idx in range(self.size()):
                if self._heap[idx][0] >= key:
                    self._heap.insert(idx, (key,val))
                    return
            self._heap.append((key, val))
        else:
            self._heap = [(key,val)]

    def pop(self):
        if self.empty():
            return None
        else:
            val = self._heap[0]
            self._heap.pop(0)
            return val
    
    def size(self) -> int:
        return len(self._heap)
    
    def empty(self) -> bool:
        return self.size() == 0
    