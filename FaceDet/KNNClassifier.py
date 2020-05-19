"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.
When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.
* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
Usage:
1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.
2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.
3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.
NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

import numpy as np
import json
import cv2
import tqdm

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_json, encode_file, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir:
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    data2 = np.load(encode_file, allow_pickle=True)
    Face_dict = data2[()]
    # {face_id : ndarray(128,)}

    with open(train_json) as json_file:
	    training = json.load(json_file)

    # Loop through each person in the training set
    for i in training:
        if training[i] == 11:
            continue
        face_id = int(i)
        this_faceCode = Face_dict[face_id]
        #this_class = [0] * 12
        #this_class[training[i]] = 1
        X.append(this_faceCode)
        y.append(training[i])

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(face_id, Face_dict, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)



    # Load image file and find face locations
    #X_img = face_recognition.load_image_file(X_img_path)
    faces_encodings = [Face_dict[face_id]]

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = closest_distances[0][0][0] <= distance_threshold

    if not are_matches:
        return -1, 0

    pred = knn_clf.predict(faces_encodings)
    score = knn_clf.predict_proba(faces_encodings)[0][pred[0]]

    return pred[0] , score


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



if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    # print("Training KNN classifier...")
    # classifier = train('./training.json','./encoding.npy', model_save_path="trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images

    target = './face'
    lst = os.listdir(target)
    lst.sort(key=lambda x: int(x.split('.jpg')[0]))

    detDict = {}
    #{id : (class, score)}

    for image_file in tqdm.tqdm(lst):
        full_file_path = os.path.join("face", image_file)


        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        face_id = int(image_file.split('.jpg')[0])

        data2 = np.load('./encoding.npy', allow_pickle=True)
        Face_dict = data2[()]

        cla, score = predict(face_id,Face_dict, model_path="trained_knn_model.clf")
        detDict[face_id] = (int(cla), float(score) )


        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("face", image_file), cla, score)

    with open('detection.json', 'w') as fp:
        json.dump(detDict, fp)