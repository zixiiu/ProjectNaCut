# import the necessary packages
from threading import Thread
from multiprocessing import Process, Lock, Manager
import sys
import cv2
import time

# import the Queue class from Python 3

# from queue import Queue

# import cut detector

from Util.CutDetectior import CutDetector

# yolo v3, tf2
#from yolov3_tf2.Connections import YOLO
#from PIL import Image

from Util.updateSSIM import updateSSIM
from Util.yoloDect import yoloDetectProcess




class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=64):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.paused = False
        self.transform = transform
        # self.yolo = YOLO()

        # Queue n Value
        self.Manager = Manager()
        self.raw_frame_queue = self.Manager.Queue(maxsize=queue_size)
        self.ssim_queue = self.Manager.Queue(maxsize=queue_size)
        self.isCut_queue = self.Manager.Queue(maxsize=queue_size)
        self.yolo_queue = self.Manager.Queue(maxsize=queue_size)
        self.ssim_sync_fn = self.Manager.Value('i', 0)
        self.yolo_sync_fn = self.Manager.Value('i', 0)

        # intialize thread
        self.io_thread = Thread(target=self.update, args=())
        self.isCut_thread = Thread(target=self.updateCut, args=())
        self.io_thread.daemon = True
        self.cutDet = CutDetector(0.3)


        # init process
        self.SSIMprocessNumber = 12
        self.yoloProcessNumber = 1
        self.SSIMProc = []
        self.yoloProc = []
        for i in range(self.SSIMprocessNumber):
            self.SSIMProc.append(
                Process(target=updateSSIM, args=(self.stopped, self.raw_frame_queue, self.ssim_queue, self.ssim_sync_fn,)))
        for i in range(self.yoloProcessNumber):
            self.yoloProc.append(Process(target=yoloDetectProcess, args=(self.stopped, self.isCut_queue, self.yolo_queue, self.yolo_sync_fn)))

    def start(self):
        # start a thread to read frames from the file video stream
        self.io_thread.start()
        for i in self.SSIMProc:
            i.start()
        # for i in self.yoloProc:
        #     i.start()
        self.isCut_thread.start()
        return self

    def updateCut(self):
        while True:
            if self.stopped:
                break

            if not self.ssim_queue.empty():
                (grabbed, curr_frame, frame, score) = self.ssim_queue.get()
                # last_frame = cv2.cvtColor()
                isCut = self.cutDet.putFrame(score)

                self.isCut_queue.put((grabbed, curr_frame, frame, isCut))
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

            if not self.raw_frame_queue.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stopped = True
                    continue

                curr_frame = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))

                if self.transform:
                    frame = self.transform(frame)

                self.raw_frame_queue.put((grabbed, curr_frame, frame, last_frame))
                last_frame = frame
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        # print(self.Q.qsize())
        return self.isCut_queue.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.raw_frame_queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.raw_frame_queue.qsize() > 0

    def pause(self):
        self.paused = True

    def cont(self):
        self.paused = False

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.io_thread.join()

    def getFrameCount(self):
        return self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
