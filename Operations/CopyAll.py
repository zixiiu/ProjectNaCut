import os
import cv2
import subprocess
import shutil
import tqdm

base_dir = "D:\Testv\TESTV官方频道"
tar_dir = "D:\Testv\VideoOnly"

todo_list = []
merge_list = set()
vid = 0
for album in tqdm.tqdm(os.listdir(base_dir)):
    album_path = os.path.join(base_dir,album)
    cid = 0
    listDirList = os.listdir(album_path)
    listDirList.sort()
    for video in listDirList:
        video_path = os.path.join(album_path,video)
        if video.endswith(".flv"):
            name = str(vid) + "_" + str(cid) + "_old.flv"
            cid += 1
            shutil.copyfile(os.path.join(album_path, video), os.path.join(tar_dir, name))
        elif video.endswith(".mp4"):
            name = str(vid) + "_" + str(cid) + "_new.mp4"
            cid += 1
            shutil.copyfile(os.path.join(album_path,video), os.path.join(tar_dir,name))

    vid += 1

