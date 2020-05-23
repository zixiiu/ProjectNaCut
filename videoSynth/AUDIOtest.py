# import modules
import madmom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import librosa
#import IPython.display as ipd

x, sr = librosa.load('ccw.wav')

proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()('ccw.wav')
# proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100, threshold=0.9)
# act = madmom.features.onsets.RNNOnsetProcessor()('ccw.wav')
beat_times = proc(act)

# onset_frames = librosa.onset.onset_detect(y=x, sr=sr)
# beat_times = librosa.frames_to_time(onset_frames, sr=sr)


print(beat_times)
clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
librosa.output.write_wav('withclick.wav', x + clicks, sr, norm=False)