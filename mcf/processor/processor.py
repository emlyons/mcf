from mcf.processor.processor_status import ProcessorStatus
class Processor:

    def __init__(self):
        self.configured = False
    
    def configure(self) -> ProcessorStatus:
        self.configured = True
        return ProcessorStatus.ERROR_NOT_IMPLEMENTED
    
    def process_frame(self, frame) -> ProcessorStatus:
        status = ProcessorStatus.SUCCESS

        if not self.configured:
            status = ProcessorStatus.ERROR_NOT_CONFIGURED
            
        if status == ProcessorStatus.SUCCESS:
            status = ProcessorStatus.ERROR_NOT_IMPLEMENTED

        return status
    