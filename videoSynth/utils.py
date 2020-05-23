import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def getPathForVideo(v):
	return v.path.replace('/media/seb101-user/New Volume/Testv2/', 'E:/Testv2/')


def getStartStop(person, length):
	total_length = person.end_frame_no - person.start_frame_no
	if length > total_length:
		return (None, None)
	keyFrame = person.keyface.at_frame
	if keyFrame + length // 2 >= person.end_frame_no:
		return (person.end_frame_no - length, person.end_frame_no)
	elif keyFrame - length // 2 <= person.start_frame_no:
		return (person.start_frame_no, person.start_frame_no + length)
	else:
		return (keyFrame - length // 2, keyFrame - length // 2 + length)


def DominateFilter(al, perList):
	p2 = []
	for p in tqdm.tqdm(perList, desc="DCF"):
		pic = al.queryPersonByCut(p.cut.id)
		max_id = -1
		max_area = 0
		for pi in pic:
			f = pi.keyface
			if f is None:
				continue
			thisarea = (f.x2 - f.x1) * (f.y2 - f.y1)
			if thisarea >= max_area:
				max_area = thisarea
				max_id = pi.id

		if pi.id == p.id:
			p2.append(p)

	return p2


def faceSizeFilter(per, threshold):
	p2 = []
	for p in per:
		f = p.keyface
		area = (f.x2 - f.x1) * (f.y2 - f.y1)
		if area >= threshold:
			p2.append(p)

	return p2


def change_cv2_draw(image,strs,local,sizes,colour):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("c:/Windows/fonts/msyh.ttc",sizes, encoding="utf-8")
    shadow = 3
    draw.text((local[0] -shadow, local[1]), strs, (0,0,0), font=font)
    draw.text((local[0] +shadow, local[1]), strs, (0,0,0), font=font)
    draw.text((local[0], local[1] + shadow), strs, (0,0,0), font=font)
    draw.text((local[0], local[1] - shadow), strs, (0,0,0), font=font)

    draw.text(local, strs, colour, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image
