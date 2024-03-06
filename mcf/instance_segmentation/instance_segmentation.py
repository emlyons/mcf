from ultralytics import YOLO
import torch

class InstanceSegmentation:

    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
    def run(self, image):
        with torch.no_grad():
            results = self.model(image, device=self.device, stream=True, verbose=False)