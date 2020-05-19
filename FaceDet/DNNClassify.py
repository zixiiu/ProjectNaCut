import os
import tqdm
import numpy as np
import cv2
import tensorflow.keras.models as k
import json

def show_prediction_labels_on_image(img_path, pred, score):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    rela = {0: 'nvhouqi',
            1: 'nanhouqi',
            2: 'hanhan',
            3: 'shushu',
            4: 'qunyan',
            5: 'laoban',
            6: 'baindaoBF',
            7: 'xinyue',
            8: 'biandao',
            9: 'py',
            10: 'xiaoshuaige',
            11: 'other',
            -1: 'unknown'}


    image = cv2.imread(img_path)

    if image.shape[0] < 160 or image.shape[1] < 160:
        return
    if pred == 0:
        color = (0,0,255)
    else:
        color = (255,255,255)
    cv2.putText(image, rela[pred] + '-' + str(score), (0,100), thickness=5, color = color, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2)
    cv2.imshow('img', image)
    cv2.waitKey(1)

target = './face'
lst = os.listdir(target)
lst.sort(key= lambda x: int(x.split('.jpg')[0]))

detDict = {}


model = k.load_model('./model86')

face_reserve = []
b_size = 64

data2 = np.load('./encoding.npy', allow_pickle=True)
Face_dict = data2[()]

for image_file in tqdm.tqdm(lst):
	full_file_path = os.path.join("face", image_file)

	# Find all people in the image using a trained classifier model
	# Note: You can pass in either a classifier file name or a classifier model instance
	face_id = int(image_file.split('.jpg')[0])
	# if face_reserve < b_size:
	# 	face_reserve.append(face_id)
	# 	continue

	thisFace_encode = np.asarray([Face_dict[face_id]])
	res = model.predict(thisFace_encode)
	score = res.max()
	cla = np.argmax(res[0])
	detDict[face_id] = (int(cla), float(score))

	# Display results overlaid on an image
	show_prediction_labels_on_image(os.path.join("face", image_file), cla, score)

with open('DNNdetection.json', 'w') as fp:
	json.dump(detDict, fp)