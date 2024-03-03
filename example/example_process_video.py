import argparse
import cv2 as cv
from mcf import Api as mcf_api

def main(input_path):
    video = cv.VideoCapture(input_path)

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv.CAP_PROP_FPS))

    mcf = mcf_api()

    while True:
        success, frame = video.read()

        if success:

            mcf.add_frame(frame)

            out_frame = mcf.next_result()

            cv.imshow(f'{width}x{height}, duration: {frames/fps}s  @ {fps}fps', frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            print('end of video')
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Python Program")
    parser.add_argument('-i','--input', dest='input', help="path to .mp4 file", required=True)
    args = parser.parse_args()

    main(args.input)