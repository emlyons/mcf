import numpy as np
from mcf.processor.processor_status import ProcessorStatus
from mcf.data_store import DataStore, DataStoreStatus
from mcf.common.time_stamp import TimeStamp
from mcf.common.queue import Queue
class Processor:

    def __init__(self):
        self.data_store = DataStore()
        self.keys = Queue()

    def process_frame(self, frame: np.array) -> ProcessorStatus:
        status = ProcessorStatus.SUCCESS

        time_stamp = TimeStamp.make()
        self.keys.push(time_stamp)

        if self.data_store.put(time_stamp, frame) != DataStoreStatus.SUCCESS:
            status = ProcessorStatus.ERROR_INTERNAL

        return status
    
    def _pipeline(self):
        # get current and last -> remove last if it exists
        
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

        return
    