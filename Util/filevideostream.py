# import the necessary packages
from threading import Thread
from multiprocessing import Process, Queue, Value, Lock, Manager
import sys
import cv2
import time

# import the Queue class from Python 3

# from queue import Queue

# import cut detector

from Util.CutDetectior import CutDetector

# yolo v3, tf2
from yolov3_tf2.Connections import YOLO
from PIL import Image

from skimage.measure import compare_ssim


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


class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=64):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.paused = False
        self.transform = transform
        # self.yolo = YOLO()

        # initialize the queue used to store frames read from
        # the video file
        self.Manager = Manager()
        self.FrameQueue = self.Manager.Queue(maxsize=queue_size)
        self.CutQueue = self.Manager.Queue(maxsize=queue_size)
        self.takeQueue = self.Manager.Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())

        self.lastFrameNumber = self.Manager.Value('i', 0)
        self.lock = Lock()

        self.thread.daemon = True
        self.cutDet = CutDetector(0.3)
        self.processNumber = 8
        self.cutProc = []
        self.takeProc = Thread(target=self.updateCut, args=())
        for i in range(self.processNumber):
            self.cutProc.append(
                Process(target=updateSSIM, args=(self.stopped, self.FrameQueue, self.CutQueue, self.lastFrameNumber, self.lock,)))

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        for i in self.cutProc:
            i.start()
        self.takeProc.start()
        return self

    def updateCut(self):
        while True:
            if self.stopped:
                break

            if not self.CutQueue.empty():
                (grabbed, curr_frame, frame, score) = self.CutQueue.get()
                # last_frame = cv2.cvtColor()
                isCut = self.cutDet.putFrame(score)

                self.takeQueue.put((grabbed, curr_frame, frame, isCut))
            else:
                time.sleep(0.1)

    def update(self):
        last_frame = None
        while True:
            if self.stopped:
                break
            if self.paused:
                time.sleep(0.1)
                continue

            if not self.FrameQueue.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stopped = True
                    continue

                curr_frame = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))

                if self.transform:
                    frame = self.transform(frame)

                self.FrameQueue.put((grabbed, curr_frame, frame, last_frame))
                last_frame = frame
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        # print(self.Q.qsize())
        return self.takeQueue.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.FrameQueue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.FrameQueue.qsize() > 0

    def pause(self):
        self.paused = True

    def cont(self):
        self.paused = False

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()

    def getFrameCount(self):
        return self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
