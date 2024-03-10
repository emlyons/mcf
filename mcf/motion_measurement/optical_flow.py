import numpy as np
import cv2 as cv

def opticalFlow(frame, frame_last, block_size):
    # frame_out = frame.copy().astype('uint8')
    meta_data = np.zeros((frame.shape[0], frame.shape[1], 2))

    for y in range(block_size[0]//2, frame.shape[0]-block_size[0]//2, block_size[0]):
        for x in range(block_size[0]//2, frame.shape[1]-block_size[1]//2, block_size[1]):
            ylow = y - block_size[0]//2
            yhigh = y + block_size[0]//2+1

            xlow = x - block_size[1]//2
            xhigh = x + block_size[1]//2+1

            # dx + dy = -dt
            block_y1 = frame[ylow:yhigh, xlow:xhigh]
            block_y3 = frame[ylow+1:yhigh+1, xlow:xhigh]
            dy = (block_y3 - block_y1)/2

            block_x1 = frame[ylow:yhigh, xlow:xhigh]
            block_x3 = frame[ylow:yhigh, xlow+1:xhigh+1]
            dx = (block_x3 - block_x1)/2

            block_1 = frame[ylow:yhigh, xlow:xhigh]
            block_2 = frame_last[ylow:yhigh, xlow:xhigh]

            dt = block_2 - block_1

            IxIx = np.sum(np.power(dx,2))
            IyIy = np.sum(np.power(dy,2))
            IxIy = np.sum(dx*dy)
            IxIt = np.sum(dx*dt)
            IyIt = np.sum(dy*dt)

            #Ax = B
            # x = inv(AtA)AtB
            AtA = np.array([[IxIx, IxIy], [IxIy, IyIy]])
            AtB = np.array([[IxIt], [IyIt]])
            uv = np.linalg.inv(AtA) @ AtB

            start_point = np.array([x-int(uv[0][0]*20), y-int(uv[1][0]*20)])
            end_point = np.array([x, y])
            
            

            # magnitude = np.linalg.norm(np.array(end_point) - np.array(start_point))
            # if magnitude > 10:
            #     frame_out = cv.arrowedLine(frame_out[:,:,], (start_point), (end_point), (25,255,25), 2)
            #     meta_data[ylow:yhigh, xlow:xhigh] = np.array([end_point[0]-start_point[0], end_point[1]-start_point[1]])

    return None, meta_data