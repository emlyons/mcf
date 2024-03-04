from mcf.processor import Processor, ProcessorStatus

class Api:

    def __init__(self):
        self.processor = Processor()
    
    def add_frame(self, frame):
        self.processor.process_frame(frame)
    
    def next_result(self):
        return None
    