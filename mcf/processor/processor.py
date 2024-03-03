from mcf.processor.processor_status import ProcessorStatus
class Processor:

    def __init__(self):
        return
    
    def process_frame(self, frame) -> ProcessorStatus:
        status = ProcessorStatus.SUCCESS
            
        if status == ProcessorStatus.SUCCESS:
            status = ProcessorStatus.ERROR_NOT_IMPLEMENTED

        return status
    