import os
import cv2

base_dir = "D:\Testv\TESTV官方频道"

todo_list = []
for album in os.listdir(base_dir):
    album_path = os.path.join(base_dir,album)
    vid_list = []
    for video in os.listdir(album_path):
        video_path = os.path.join(album_path,video)
        if video.endswith(".mp4"):
            vid_list.append(video_path)
        if video.endswith(".aria2"):
            print(album)
            #skip album
            break
    todo_list += vid_list

print(todo_list)
tot_frame = 0
d = 0
for vid in todo_list:
    cap = cv2.VideoCapture(vid)
    time = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)/60
    if time < 7:
        print(str(time)+"  "+vid)
        d += 1
    tot_frame += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(tot_frame)
print(d)