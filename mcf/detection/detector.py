from ultralytics import YOLO
import torch
from mcf.detection.detection_status import DetectionStatus

class Detector:

    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')

        if torch.cuda.is_available():
            self.device = 'cuda'
        # elif torch.backends.mps.is_available():
        #     self.device = 'mps'
        else:
            self.device = 'cpu'
        
    def run(self, image):
        status = DetectionStatus.SUCCESS
        
        result = self._run_model(image)



        return status, result
    
    def _run_model(self, image):
        with torch.no_grad():
            output = self.model(image, device=self.device, stream=True, verbose=False)
            result = [r for r in output]
            result = result[0]

        return result