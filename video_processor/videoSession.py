from __future__ import division, print_function, absolute_import

import Util.filevideostream as filevideostream

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolov3_tf2.Connections import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import Util.CutDetectior
import face_recognition


class videoSession(object):
    def __init__(self, videoPath, visualize = False):
        self.visualizeFlag = visualize
        # Yolo
        self.yolo = YOLO()
        # Definition of the parameters
        max_cosine_distance = 0.3
        nn_budget = None
        self.nms_max_overlap = 1.0

        # deep_sort
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        self.video_capture = filevideostream.FileVideoStream(videoPath)

        #cutID
        self.cut_id = 0

    def start(self):
        self.video_capture.start()

    def visualize(self, inDict, frame):
        cv2.putText(frame, 'frame %d, cut%d'%(inDict['frame_no'],inDict['cut_id']),(0,0),0,5e-3 * 200, (0,255,0),2)
        for people in inDict['person']:
            cv2.rectangle(frame, (people['x1'], people['y1']), (people["x2"], people['y2']), (255,255,255), 2)
            if people['face']:
                x1 = people['x1'] + people['face']['x1']
                x2 = people['x2'] + people['face']['y1']
                y1 = people['y1'] + people['face']['x2']
                y2 = people['y2'] + people['face']['y2']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('', frame)

    def nextFrame(self):
        ret, frame_no, frame, isCut = self.video_capture.read()
        resDict = {}
        if ret != True:
            return None

        # get yolo boxes
        boxs = self.yolo.detect_image(frame)
        # print("box_num",len(boxs))
        features = self.encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        if isCut:
            self.tracker.delete_all()
            self.cut_id += 1
        self.tracker.predict()
        self.tracker.update(detections)

        resDict['frame_no'] = frame_no
        resDict['is_cut'] = isCut
        resDict['cut_id'] = self.cut_id
        resDict['person'] = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            pifDict = {'trackId': track.track_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            peopleFrame = frame[y1:y2, x1:x2]
            faceDict = None
            face_locations = face_recognition.face_locations(peopleFrame)
            for top, right, bottom, left in face_locations:
                faceDict = {'x1': left, 'x2': right, 'y1': top, 'y2': bottom}
                break
            pifDict['face'] = faceDict
            resDict['person'].append(pifDict)
        if self.visualizeFlag:
            self.visualize(resDict,frame)
        return resDict
