from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from video_processor.ORMModel import *
from video_processor.dbAllocator import Allocator
import tqdm
import Util.Purefilevideostream as cv
import face_recognition
import cv2
import os

engine = create_engine('sqlite:///testv.sqlite')
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
totalPIFNo = 0
while start < totalPIF:
	PIF = session.query(PersonInFrame).filter(PersonInFrame.id > start).limit(step).all()
	for p in PIF:
		if p.id not in hasFacePIFSet: #does not have face
			Vid = p.video_id
			totalPIFNo += 1
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

#print(cropDict)

al = Allocator('sqlite:///testv.sqlite', '/media/seb101-user/New Volume/')
detectFrame = []
pbar2.close()
pbar3 = tqdm.tqdm(desc='total:', total = totalPIFNo)
frameToRun = []
PIFIdInList = []

for video in session.query(Video).all():
	if video.id in faceVSet:
		pbar3.update(cropDict[video.id].__len__())
		pbar3.refresh()
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


	#print(video.path)
	h = cv.FileVideoStream(video.path)
	h.start()
	while True:
		fn, ret = h.read()
		if ret is None:
			break

		if fn in thisCropDict:
			for box in thisCropDict[fn]:
				#(x1, y1, x2, y2, PIFID, at_frame)
				#cv2.imwrite(os.path.join('./PIFforFace', str(box[4]) + '.jpg'))
				#face_locations = face_recognition.face_locations(cropped, model='cnn')
				cropped = ret[box[1]:box[3], box[0]:box[2]].copy()
				frameToRun.append(cropped)
				PIFIdInList.append(box[4])
				#cv2.imshow('PIF', cropped)
				#cv2.waitKey(1)
				pbar3.update(1)

				if len(frameToRun) == 32:

					batch_of_face_locations = face_recognition.batch_face_locations(frameToRun,
					                                                                number_of_times_to_upsample=0)

					for i in range(len(frameToRun)):
						PIFId = PIFIdInList[i]
						face_loc = batch_of_face_locations[i]

						for top, right, bottom, left in face_loc:
							faceDict = {'x1': left, 'x2': right, 'y1': top, 'y2': bottom}
							al.writeFace(left,top,right,bottom,PIFId)
							break
					frameToRun = []
					PIFIdInList = []







