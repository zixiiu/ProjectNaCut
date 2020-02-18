#! /usr/bin/env python
# -*- coding: utf-8 -*-

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


import os

import face_recognition



def main():
    yolo = YOLO()
   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False
    doFace_flag = True

    video_capture = filevideostream.FileVideoStream("./testVideo/604_0_new.mp4")
    video_capture.start()
        # cv2.VideoCapture("/media/seb101-user/DATA/TestV_videos/447_1_old.mp4")

    #CutDetector
    #cutDetector = Util.CutDetectior.CutDetector(threshold=0.3)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        print(w)
        print(h)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    #yolo multiprocess queues/values

    while True:
        tfps = time.time()
        tget = time.time()
        ret, frame_no, frame, isCut = video_capture.read()
        if ret != True:
            break
        tget = time.time() - tget


        # isCut = cutDetector.putFrame(frame)
        # people_frame = []
        #face_locations = face_recognition.face_locations(frame, model="cnn")
       # image = Image.fromarray(frame)
        #image = Image.fromarray(frame[...,::-1]) #bgr to rgb


        tyolo = time.time()
        boxs = yolo.detect_image(frame)
        # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        tyolo = time.time() - tyolo

        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        ttrack = time.time()
        if isCut:
            tracker.delete_all()
        tracker.predict()
        tracker.update(detections)
        ttrack = time.time() - ttrack


        #draw image
        tface = time.time()
        for det in detections:
            bbox = det.to_tlbr()
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            peopleFrame = frame[y1:y2, x1:x2]
            if doFace_flag:
                face_locations = face_recognition.face_locations(peopleFrame, model='cnn')
                for top, right, bottom, left in face_locations:
                    top += y1
                    bottom += y1
                    right += x1
                    left += x1
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        tface = time.time() - tface

        tvis = time.time()
        for det in detections:
            bbox = det.to_tlbr()
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0])+50, int(bbox[1])+50),0, 5e-3 * 200, (0,255,0),2)





        cv2.imshow('', frame)
        tvis = time.time() - tvis
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = 1/(time.time()- tfps)
        print("fps= %.2f, frame:%0.f, tget:%.2f tyolo:%.2f, ttrack:%.2f, tface:%.2f, tvis: %.2f"%(fps, frame_no, tget*1000, tyolo*1000, ttrack*1000,tface * 1000, tvis*1000))
        if isCut:
            print("==============================================================================================")
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.stop()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
