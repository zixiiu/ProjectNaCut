from video_processor import dbAllocator
import tqdm
from Util import Purefilevideostream
import cv2
import os
import utils
import random

al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')
per = al.queryAllPerson()
per = [i for i in per if i.name_mark == 'nvhouqi' and i.name_conf >= 960]

# Dominating Character Filter
per = utils.DominateFilter(per)


random.shuffle(per)
writer = cv2.VideoWriter('res.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))
ind = 0
for p in tqdm.tqdm(per):
	ind += 1
	if ind == 300:
		break
	path = utils.getPathForVideo(p.video)
	videoHandle = cv2.VideoCapture(path)
	start, end = utils.getStartStop(p, 60)
	if start is None:
		continue
	videoHandle.set(cv2.CAP_PROP_POS_FRAMES, start)
	for i in range(end-start):
		grab, frame = videoHandle.read()
		writer.write(frame)
	videoHandle.release()

writer.release()
print(10)
