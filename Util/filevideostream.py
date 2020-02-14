# import the necessary packages
from threading import Thread
#from multiprocessing import Process, Queue
import sys
import cv2
import time

# import the Queue class from Python 3

from queue import Queue

# import cut detector

from Util.CutDetectior import CutDetector

# yolo v3, tf2
from yolov3_tf2.Connections import YOLO
from PIL import Image


class FileVideoStream:
	def __init__(self, path, transform=None, queue_size=64):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		self.paused = False
		self.transform = transform
		#self.yolo = YOLO()

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queue_size)
		# intialize thread
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.cutDet = CutDetector(0.3)

	def start(self):
		# start a thread to read frames from the file video stream
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				break

			if self.paused:
				time.sleep(0.1)  # Rest for 10ms, we don't need update
				continue

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				if not grabbed:
					self.stopped = True
					continue

				# read and the frame number
				curr_frame = int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))

				# cut det
				isCut = self.cutDet.putFrame(frame)
				#isCut = False

				# yolo
				#boxs = self.yolo.detect_image(frame)




				if self.transform:
					frame = self.transform(frame)

				# add the frame to the queue
				self.Q.put((grabbed, curr_frame, frame, isCut))
			else:
				time.sleep(0.1)  # Rest for 10ms, we have a full queue

		self.stream.release()

	def read(self):
		# return next frame in the queue
		#print(self.Q.qsize())
		return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
	def running(self):
		return self.more() or not self.stopped

	def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
		tries = 0
		while self.Q.qsize() == 0 and not self.stopped and tries < 5:
			time.sleep(0.1)
			tries += 1

		return self.Q.qsize() > 0

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
