import numpy as np
from mcf.processor.processor_status import ProcessorStatus
from mcf.data_store import DataStore, DataStoreStatus
from mcf.frame import Frame
from mcf.common.queue import Queue
from mcf.display import Display

class Processor:

    def __init__(self, enable_display):
        self.data_store = DataStore()
        self.queue = Queue()
        self.enable_display = enable_display
        self.display = Display()

    def process_frame(self, image: np.array) -> ProcessorStatus:
        frame = Frame(image=image)
        self.queue.push(frame)
        status = self._pipeline()
        return status
    
    def _pipeline(self) -> ProcessorStatus:
        # get current and last -> remove last if it exists
        frame = self.queue.front()
        self.queue.pop()
        
        # YOLO: Classification/Segmentation (current)

        # proceed if last exists

        # classifier filtering w/ recursive bayes (current, last)

        # motion measurement w/ optical flow (current, last)

        # motion prediction model (current, last)

        # region matching, associate detections between images based on predictions to minmize error (current, last)

        # filtering w/ kalman or partical filter for measurement and prediction resolution (current, last)

        # motion based degredation model

        # image recovery

        # display
        if self.enable_display:
            self.display.show(frame.image)

        return
    