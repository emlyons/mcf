from mcf.processor import Processor, ProcessorStatus

class Api:

    def __init__(self, enable_display=False):
        self.processor = Processor(enable_display)
    
    def add_frame(self, frame):
        self.processor.process_frame(frame)
    
    def next_result(self):
        return None
    