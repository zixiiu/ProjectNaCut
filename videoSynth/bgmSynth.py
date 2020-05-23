import madmom
import librosa
from video_processor import dbAllocator
import cv2
import random
import utils
import tqdm
import json
from PIL import Image

# bgm beat detection
audioFile = 'mine.wav'
#x, sr = librosa.load(audioFile)

proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(audioFile)
# proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100, threshold=0.9)
# act = madmom.features.onsets.RNNOnsetProcessor()(audioFile)
beat_times = proc(act)


# normalize to 30 fps
keyFrame = []
for i in beat_times:
	keyFrame.append(int(i // 0.033) + 1)

# video//

al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')
per = al.queryAllPerson()
per = [i for i in per if i.name_mark == 'nvhouqi' and i.name_conf >= 960]
per = utils.DominateFilter(al, per)
per = utils.faceSizeFilter(per, 20000)

r_seed = random.Random(114514)
r_seed.shuffle(per)
writer = cv2.VideoWriter('res.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))

# title Dict
with open('videoTitle.json') as json_file:
	titleDict = json.load(json_file)

# block list
bl = [63581, 95644, 102822, 131197, 119360, 23217, 55347, 111391, 92226, 151164, 24060, 132529, 2105, 63701, 103418,
      145086, 23161, 169512, 123894, 151171, 147061, 100560, 108931, 131015, 8059, 68207, 112131, 124036, 11254, 44002,
      76636, 65361, 89135, 8062,133510, 111052, 29309, 133248, 110443, 121164, 121384, 144269, 142839, 123208,143259, 135812,
      84209, 192343, 94153, 177953, 44830, 162779, 133262, 160975, 193577, 84259, 94029, 61816, 24534, 23134, 123207, 93920,
      106359, 137672, 159601, 42439, 107397, 68071, 160973,34347, 23116, 120037, 60885, 150348, 92578, 128627, 190547 ]
#122143, too long
bl = set(bl)

# Skip Chance
sc = 0.5
minStack = 2
maxStack = 4
# start synth
print(len(keyFrame))
startM = 0
endM = 0
stack = 0
cutCount = 0
for i in tqdm.tqdm(keyFrame):
	endM = i
	stack += 1
	if r_seed.random() < sc and stack < maxStack:
		# print('skip!')
		continue
	# if endM - startM < 30:
	# 	print('next loop')
	# 	continue

	start = None
	while start is None:
		p = per.pop()
		# per.append(p)
		if p.id in bl:
			continue
		start, end = utils.getStartStop(p, endM - startM)

	videoHandle = cv2.VideoCapture(utils.getPathForVideo(p.video))

	videoHandle.set(cv2.CAP_PROP_POS_FRAMES, start)
	for i in range(end - start):
		grab, frame = videoHandle.read()
		vid = p.video.avid
		time = int(round(start / videoHandle.get(cv2.CAP_PROP_FPS)))
		timestr = str(time // 60) + ':' + str(time % 60).zfill(2)
		wd = 'av' + str(vid) + ': ' + titleDict[str(vid)].split('_')[0]
		frame = utils.change_cv2_draw(frame, wd, (100, 980), 30, (255, 255, 255))
		frame = utils.change_cv2_draw(frame, timestr, (100, 1020), 30, (255, 255, 255))
		# frame = utils.change_cv2_draw(frame, str(p.id), (300, 1020), 30, (255, 255, 255))

		# cv2.imshow('wind', frame)
		# cv2.waitKey(1)
		writer.write(frame)
	cutCount += 1
	videoHandle.release()

	stack = 0
	startM = endM

writer.release()
print('total cut:', cutCount)
# ffmpeg -i res2.mp4 -i mine.wav -c:v libx264 -c:a aac
