from queue import Queue as pyQueue

class Queue:

    def __init__(self):
        self._queue = pyQueue()
    
    def push(self, val):
        self._queue.put(val)

    def pop(self):
        if self._queue.qsize() > 0:
            self._queue.get()

    def front(self):
        value = None
        if self._queue.qsize() > 0:
            value = self._queue.queue[0]
        return value
    
    def size(self) -> int:
        return self._queue.qsize()
    
    def empty(self) -> bool:
        return self.size() == 0
    