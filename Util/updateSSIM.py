from skimage.metrics import structural_similarity
import time
import cv2

def updateSSIM(stopped, FrameQueue, CutQueue, lastFrameNumber):
    while True:
        if stopped:
            break

        if not FrameQueue.empty():
            (grabbed, curr_frame, frame, last_frame) = FrameQueue.get()
            score = 0
            if last_frame is not None:
                # last_frame = cv2.cvtColor()
                last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = structural_similarity(last_frame, frame_gray, gradient=False, full=False, multichannel=False)

            while lastFrameNumber.value != curr_frame - 1:
                time.sleep(0.01)

            CutQueue.put((grabbed, curr_frame, frame, score))
            lastFrameNumber.value = curr_frame

        else:
            time.sleep(0.1)