from video_processor import dbAllocator
from video_processor.ORMModel import *
import cv2
import utils



al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')
for i in al.session.query(Video).all():
	path = utils.getPathForVideo(i)
	vh = cv2.VideoCapture(path)
	print(vh.get(cv2.CAP_PROP_FRAME_WIDTH), vh.get(cv2.CAP_PROP_FRAME_HEIGHT))