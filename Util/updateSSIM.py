from skimage.measure import compare_ssim
import time
import cv2

def updateSSIM(stopped, FrameQueue, CutQueue, lastFrameNumber,lock):
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
                score = compare_ssim(last_frame, frame_gray, gradient=False, full=False, multichannel=True)

            while lastFrameNumber.value != curr_frame - 1:
                time.sleep(0.01)
            lock.acquire()

            CutQueue.put((grabbed, curr_frame, frame, score))
            lastFrameNumber.value = curr_frame
            lock.release()
        else:
            time.sleep(0.1)