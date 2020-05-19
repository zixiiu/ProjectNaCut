import face_recognition
import os
import tqdm

import json
import numpy as np

from multiprocessing import Queue, Process



target = './face'
lst = os.listdir(target)
lst.sort(key= lambda x: int(x.split('.jpg')[0]))

resDict = {}


for f in tqdm.tqdm(lst):

	i = face_recognition.load_image_file(os.path.join(target,f))
	# top, right, bottom, left
	en = face_recognition.face_encodings(i, known_face_locations=[[0, i.shape[1], i.shape[0], 0]], num_jitters=100)
	for i in en:
		resDict[int(f.split('.jpg')[0])] = i
		break


np.save('encoding.npy', resDict)
