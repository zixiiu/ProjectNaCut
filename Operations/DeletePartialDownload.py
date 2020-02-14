import os
import cv2
import shutil

base_dir = "D:\Testv\TESTV官方频道"

todo_list = []
for album in os.listdir(base_dir):
    album_path = os.path.join(base_dir,album)
    vid_list = []
    for video in os.listdir(album_path):
        video_path = os.path.join(album_path,video)
        if len(os.listdir(album_path)) == 1:
            print("Delete: " + album_path)
            shutil.rmtree(album_path)
        # if video.endswith(".mp4"):
        #     vid_list.append(video_path)
        #     cap = cv2.VideoCapture(video_path)
        #     #print(cap.get(cv2.CAP_PROP_FPS))
        #     time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) / 60
        #     cap.release()
        #     if time < 8:
        #         print("Delete: " + album_path)
        #         shutil.rmtree(album_path)
        if video.endswith(".aria2"):
            print("Delete: " + album_path)
            shutil.rmtree(album_path)

            break
    todo_list += vid_list

