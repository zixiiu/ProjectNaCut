import cv2
import os
import json
import random

target = './face'
lst = os.listdir(target)
lst.sort(key= lambda x: int(x.split('.jpg')[0]))

res = {}

cont = True

rela = {0:'女后期',
1:'男后期',
2:'憨憨',
3:'叔叔',
4:'群演',
5:'老板',
6:'编导男朋友',
7:'心悦',
8:'编导',
9:'配音君',
10:'小帅哥',
11:'other'}
labeled = 0

lastloc = 0
loadLast = False

while True:
	if loadLast:
		loc = lastloc
	else:
		loc = random.randint(0,len(lst))
	i = lst[loc]
	face_id = int(i.split('.jpg')[0])


	if face_id in res:
		continue

	if not cont:
		break

	img = cv2.imread(os.path.join(target, i))
	if img.shape[0] < 160 or img.shape[1]< 160:
		continue
	print(labeled)
	cv2.imshow('img',img)
	labeled += 1

	lastloc = loc
	while True:
		k = cv2.waitKey(33)

		if k == ord('z'):
			loadLast = True
			break
		if k == ord('`'):
			res[face_id] = 0
			break
		if k == ord('1'):
			res[face_id] = 1
			break
		if k == ord('2'):
			res[face_id] = 2
			break
		if k == ord('3'):
			res[face_id] = 3
			break
		if k == ord('4'):
			res[face_id] = 4
			break
		if k == ord('5'):
			res[face_id] = 5
			break
		if k == ord('6'):
			res[face_id] = 6
			break
		if k == ord('7'):
			res[face_id] = 7
			break
		if k == ord('8'):
			res[face_id] = 8
			break
		if k == ord('9'):
			res[face_id] = 9
			break
		if k == ord('0'):
			res[face_id] = 10
			break
		if k == ord('o'):
			res[face_id] = 11
			break
		if k == ord('q'):
			cont = False
			break

print(res)

with open('training.json', 'w') as fp:
    json.dump(res, fp)