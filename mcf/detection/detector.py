import torch
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from mcf.detection.detection_status import DetectionStatus
from mcf.data_types import DetectionRegion
from mcf.common import get_center_of_mass, format_mask

class Detector:

    def __init__(self):
        self._model = YOLO('yolov8n-seg.pt')

        if torch.cuda.is_available():
            self._device = 'cuda'
        # elif torch.backends.mps.is_available():
        #     self._device = 'mps'
        else:
            self._device = 'cpu'
        
    def run(self, image) -> tuple[DetectionStatus, list[DetectionRegion]]:
        output = self._run_model(image)
        status, detection_regions = self._get_detection_regions(output, image)
        return status, detection_regions
    
    def _run_model(self, image) -> torch.tensor:
        with torch.no_grad():
            output = self._model.predict(image, device=self._device, stream=True, verbose=False)
            result = [r for r in output]
            result = result[0]
        return result
    
    def _get_detection_regions(self, model_output: torch.tensor, image: np.array) -> tuple[DetectionStatus, list[DetectionRegion]]:
        status = DetectionStatus.SUCCESS

        if model_output.masks is None:
            status = DetectionStatus.EMPTY_FRAME

        detection_regions = []
        if status == DetectionStatus.SUCCESS:
            for (box, mask) in zip(model_output.boxes, model_output.masks):
                class_id = int(box.cls.item())
                ulx,uly,lrx,lry = np.array(box.xyxy.tolist()[0]).astype('int')
                bounding_box = ((ulx, uly), (lrx, lry))
                    
                mask = mask.cpu().numpy().data[0]
                mask = cv.resize(mask, (image.shape[1], image.shape[0]), cv.INTER_LINEAR)[uly:lry, ulx:lrx]
                center_of_mass = get_center_of_mass(mask)

                mask = format_mask(mask, bounding_box)

                class_id = box.cls.item()
                confidence = box.conf.item()
                detection_regions.append(DetectionRegion(class_id, confidence, bounding_box, mask, center_of_mass))

        return status, detection_regions

    