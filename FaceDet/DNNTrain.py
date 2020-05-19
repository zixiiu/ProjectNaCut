from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from keras import applications

import numpy as np
import json

data2=np.load('encoding.npy', allow_pickle=True)
Face_dict = data2[()]
#{face_id : ndarray(128,)}

with open('training.json') as json_file:
    training = json.load(json_file)

x_train = []
y_train = []
x_test = []
y_test = []
train_id_set = set()

tr_no = 0
for i in training:
	face_id = int(i)
	train_id_set.add(face_id)
	this_faceCode = Face_dict[face_id]
	this_class = [0] * 12
	this_class[training[i]] = 1
	x_train.append(this_faceCode)
	y_train.append(this_class)

	tr_no += 1
	if tr_no > 949:
		break

for i in training:
	face_id = int(i)
	if face_id not in  train_id_set:
		this_faceCode = Face_dict[face_id]
		this_class = [0] * 12
		this_class[training[i]] = 1
		x_test.append(this_faceCode)
		y_test.append(this_class)


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(tr_no)

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

model = Sequential()
model.add(BatchNormalization())
model.add(Dense(256, input_dim = 128))
model.add(LeakyReLU(alpha=0.3))
model.add(BatchNormalization())

model.add(Dense(512,  activation='relu'))
model.add(BatchNormalization())

model.add(Dense(512,  activation='relu'))
model.add(BatchNormalization())

model.add(Dense(256,  activation='relu'))
model.add(BatchNormalization())

model.add(Dense(128,  activation='relu'))
model.add(BatchNormalization())

model.add(Dense(12, activation = 'sigmoid'))




# model = Sequential()
# model.add(Dense(514, input_dim = 128, activation='relu'))

# model.add(Dense(600,  activation='relu'))
# model.add(Dense(700,  activation='relu'))
# model.add(Dense(647,  activation='relu'))
# model.add(Dense(568,  activation='relu'))
# model.add(Dense(567,  activation='relu'))
# model.add(Dense(867,  activation='relu'))
# model.add(Dense(425,  activation='relu'))
# model.add(Dense(12,  activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])



overfitCallback = EarlyStopping(monitor='accuracy', min_delta=0, patience = 100, restore_best_weights=True)


model.fit(x_train, y_train, batch_size= 256, epochs = 2000, callbacks=[overfitCallback])

result_train = model.evaluate(x_train, y_train)
result_test = model.evaluate(x_test, y_test)
print('Train Acc:', result_train[1])
print('Test Acc:', result_test[1])

model.save('./model86')

