from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from video_processor.ORMModel import *
import tqdm
import Util.Purefilevideostream as cv
import cv2
import os

engine = create_engine('sqlite:///testv_NOT_COMPLETE.sqlite')
sm = sessionmaker()
sm.configure(bind=engine)
session = sm()

hasFacePIFSet = set()

totalFace = session.query(Face).count()
start = 0
step = 100
pbar = tqdm.tqdm(total=totalFace)
while start < totalFace:
	face = session.query(Face).filter(Face.id > start).limit(step).all()
	for f in face:
		hasFacePIFSet.add(f.personInFrame_id)
	start += step
	pbar.update(step)

pbar.close()
totalPIF = session.query(PersonInFrame).count()
start = 0
step = 100
cropDict = {}
# {videoID: (x1, y1, x2, y2, PIFID, at_frame)
pbar2 = tqdm.tqdm(total=totalPIF)
faceVSet = set()
while start < totalPIF:
	PIF = session.query(PersonInFrame).filter(PersonInFrame.id > start).limit(step).all()
	for p in PIF:
		if p.id not in hasFacePIFSet: #does not have face
			Vid = p.video_id
			if Vid not in cropDict:
				cropDict[Vid] = [(p.x1, p.y1, p.x2, p.y2, p.id, p.at_frame)]
			else:
				cropDict[Vid].append((p.x1, p.y1, p.x2, p.y2, p.id, p.at_frame))
		else: #have face
			Vid = p.video_id
			if Vid not in faceVSet:
				faceVSet.add(Vid)
			else:
				pass

	start += step
	pbar2.update(step)

print(cropDict)

detectFrame = []
for video in tqdm.tqdm(session.query(Video).all()):
	if video.id in faceVSet:
		continue
	#dict-ize
	thisCropDict = {}
	for c in cropDict[video.id]:
		#videoID: (x1, y1, x2, y2, PIFID, at_frame)
		if c[5] in thisCropDict:
			thisCropDict[c[5]].append(c)
		else:
			thisCropDict[c[5]] = [c]

	#load video & crop
	print(video.path)
	h = cv.FileVideoStream(video.path)
	h.start()
	while True:
		fn, ret = h.read()
		if ret is None:
			break

		if fn in thisCropDict:
			for box in thisCropDict[fn]:
				#(x1, y1, x2, y2, PIFID, at_frame)
				cropped = ret[box[1]:box[3], box[0]:box[2]].copy()
				cv2.imwrite(os.path.join('./PIFforFace', str(box[4]) + '.jpg'))






