import torch
import numpy as np
from ultralytics import YOLO
from mcf.detection.detection_status import DetectionStatus
from mcf.data_types import DetectionRegion

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
            output = self.model.predict(image, device=self.device, stream=True, verbose=False)
            result = [r for r in output]
            result = result[0]

        return result
    
    def _get_detection_regions(self, model_output):
        status = DetectionStatus.SUCCESS

        if model_output.masks is None:
            status = DetectionStatus.EMPTY_FRAME

        detection_regions = []
        if status == DetectionStatus.SUCCESS:
            for (box, mask) in zip(model_output.boxes, model_output.masks):
                class_id = int(box.cls.item())
                ulx,uly,lrx,lry = np.array(box.xyxy.tolist()[0]).astype('int')
                bounding_box = ((ulx, uly), (lrx, lry))
                mask = mask.cpu().numpy().data.squeeze()
                detection_regions.append(DetectionRegion(class_id, None, bounding_box, mask))

        return status, detection_regions
    