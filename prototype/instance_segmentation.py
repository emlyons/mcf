import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

from ultralytics import YOLO
import torch

CAR_ID = 2

# video meta-data
# video = cv.VideoCapture(f'/Users/elyons/Documents/dev/repos/motion_compensated_filtering_for_image_recovery/prototype/data/highway.mp4')
# width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
# frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
# fps = int(video.get(cv.CAP_PROP_FPS))
# print(f'resolution: {width}x{height}, frames: {frames}, fps: {fps}, duration: {frames/fps:.3f} s')


# YOLO Configuration
model = YOLO('yolov8n-seg.pt')
# model.export(format='onnx')  # creates 'yolov8n.onnx'
# model = YOLO('yolov8n-seg.onnx') # <- onnx runtime (seems to be slower ~40ms slower, maybe it isn't using the NPU


# Run model on video data
video = cv.VideoCapture(f'/Users/elyons/Documents/dev/repos/motion_compensated_filtering_for_image_recovery/prototype/data/highway.mp4')

while True:
    success, frame = video.read()

    if success:

        results = model(frame, device='mps', stream=True, verbose=False)
        adjusted_mask = np.zeros(frame.shape, dtype='uint8')

        # # Process results list
        for result in results:

            for (box, mask) in zip(result.boxes, result.masks):
                
                # filter for cars only
                class_id = int(box.cls.item())
                if class_id == 2:

                    # add bounding box to frame
                    # x1,y1,x2,y2 = np.array(box.xyxy.tolist()[0]).astype('int')
                    # frame[y1:y2, x1:x2,1] = 255


                    # add mask
                    mask = mask.cpu().numpy().data.squeeze()
                    
                    adjusted_mask = np.zeros((frame.shape), dtype='uint8')
                    adjusted_mask[:,:,2] = cv.resize(mask, (frame.shape[1], frame.shape[0])) * 255
                    
                    frame[adjusted_mask != 0] = 255
                    # frame = cv.addWeighted(frame, 0.75, adjusted_mask, 0.25, 0)


        cv.imshow(f'instance segmentation w/ YOLOv8', frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break # end of video