import numpy as np
import cv2 as cv

# video meta-data
# video = cv.VideoCapture(f'/Users/elyons/Documents/dev/repos/motion_compensated_filtering_for_image_recovery/prototype/data/highway.mp4')
# width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
# frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
# fps = int(video.get(cv.CAP_PROP_FPS))
# print(f'resolution: {width}x{height}, frames: {frames}, fps: {fps}, duration: {frames/fps:.3f} s')

def main():
    video = cv.VideoCapture(f'/Users/elyons/Documents/dev/repos/motion_compensated_filtering_for_image_recovery/prototype/data/highway.mp4')

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv.CAP_PROP_FPS))

    while True:
        success, frame = video.read()

        if success:

            cv.imshow(f'{width}x{height}, duration: {frames/fps}s  @ {fps}fps', frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            print('end of video')
            break

if __name__ == "__main__":
    main()