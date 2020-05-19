import numpy as np
import json


with open('training.json') as json_file:
    data = json.load(json_file)

classcount = {}
for key in data:
	if data[key] not in classcount:
		classcount[data[key]] = 0
	classcount[data[key]] += 1


data2=np.load('encoding.npy', allow_pickle=True)
dict = data2[()]
print(dict)