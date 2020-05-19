from video_processor import dbAllocator
import tqdm
from Util import Purefilevideostream
import cv2
import os

al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')
target = './face'
per = al.queryAllPerson()
thisvideo_id = -1
cropDict = {}
for p in tqdm.tqdm(per):
	if thisvideo_id != p.video_id:
		#crop here!
		if thisvideo_id != -1:
			v = al.getVideoById(thisvideo_id)
			path = v.path.replace('/media/seb101-user/New Volume/Testv2/', 'E:/Testv2/')
			videoHandle = Purefilevideostream.FileVideoStream(path)
			videoHandle.start()
			while True:
				fn, ret = videoHandle.read()

				if ret is None:
					break

				if fn in cropDict:
					box = cropDict[fn]
					cropped = ret[box[1]:box[3], box[0]:box[2]].copy()

					cv2.imshow('PIF', cropped)
					cv2.imwrite(os.path.join(target, str(box[4]) + '.jpg') ,cropped)
					cv2.waitKey(1)


		cropDict = {}
		thisvideo_id = p.video_id
	if p.keyface:
		f = p.keyface
		cropDict[f.at_frame] = [f.x1_abs, f.y1_abs, f.x2_abs, f.y2_abs, f.id]

