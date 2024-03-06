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
        output = self._run_model(image)
        status, detection_regions = self._get_detection_regions(output)
        return status, detection_regions
    
    def _run_model(self, image):
        with torch.no_grad():
            output = self.model(image, device=self.device, stream=True, verbose=False)
            result = [r for r in output]
            result = result[0]

        return result
    
    def _get_detection_regions(self, model_output):
        status = DetectionStatus.SUCCESS

        if model_output.masks is None:
            status = DetectionStatus.EMPTY_FRAME

        if status == DetectionStatus.SUCCESS:
            for (box, mask) in zip(model_output.boxes, model_output.masks):
                break

        return status, None