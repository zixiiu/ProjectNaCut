from video_processor import dbAllocator
from video_processor.ORMModel import *
import cv2
import utils
import json


with open('videoTitle.json') as json_file:
    ti = json.load(json_file)

print(ti)