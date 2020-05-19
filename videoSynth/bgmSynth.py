import madmom
import librosa
from video_processor import dbAllocator
import cv2
import random
import utils
import tqdm

# bgm beat detection
x, sr = librosa.load('ccw.wav')

proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()('ccw.wav')
# proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100, threshold=0.9)
# act = madmom.features.onsets.RNNOnsetProcessor()('ccw.wav')
beat_times = proc(act)

#print(beat_times)

#normalize to 30 fps
keyFrame = []
for i in beat_times:
	keyFrame.append(int(i//0.033) + 1)


#video//

al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')
per = al.queryAllPerson()
per = [i for i in per if i.name_mark == 'nvhouqi' and i.name_conf >= 960]
per = utils.DominateFilter(al, per)
per = utils.faceSizeFilter(per, 20000)


random.shuffle(per)
writer = cv2.VideoWriter('res.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))

#start synth
startM = 0
endM = 0
for i in tqdm.tqdm(keyFrame):
	endM = i


	start = None
	while start is None:
		p = per.pop()
		start, end = utils.getStartStop(p, endM-startM)

	videoHandle = cv2.VideoCapture(utils.getPathForVideo(p.video))

	videoHandle.set(cv2.CAP_PROP_POS_FRAMES, start)
	for i in range(end - start):
		grab, frame = videoHandle.read()
		writer.write(frame)
	videoHandle.release()
	startM = endM

writer.release()

