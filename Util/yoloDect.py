from yolov3_tf2.Connections import YOLO
import time

def yoloDetectProcess(stopped, inQueue, outQueue, lastFrameNumber):
    yolo = YOLO()
    while True:
        if stopped:
            break

        if not inQueue.empty():
            (grabbed, curr_frame, frame, isCut) = inQueue.get()
            boxs = yolo.detect_image(frame)

            while lastFrameNumber.value != curr_frame - 1:
                time.sleep(0.01)

            outQueue.put((grabbed, curr_frame, frame, isCut, boxs))
            lastFrameNumber.value = curr_frame

        else:
            time.sleep(0.1)
