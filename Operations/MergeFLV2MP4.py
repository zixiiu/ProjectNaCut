import os
import cv2
import subprocess

base_dir = "D:\Testv\TESTV官方频道"

todo_list = []
merge_list = set()
for album in os.listdir(base_dir):
    album_path = os.path.join(base_dir,album)
    vid_list = []
    for video in os.listdir(album_path):
        video_path = os.path.join(album_path,video)
        if video.endswith(".flv"):
            vid_list.append(video_path)
            merge_list.add(album_path)

    todo_list += vid_list

for paths in merge_list:
    print(paths)
    mergeList = [os.path.join(paths,x) for x in os.listdir(paths) if x.endswith('.flv')]
    mp4_file = os.path.abspath('{}/{}.mp4'.format(paths, mergeList[0]))
    mergeList.sort()
    shell = 'ffmpeg -i "{}" -c copy -f mp4 -y "{}"'
    listVidStringFormat = "|".join(mergeList)
    shell.format("concat:"+listVidStringFormat, mp4_file)
    shell =  'ffmpeg -i "'+ "concat:"+listVidStringFormat +'" -c copy -f mp4 -y "'+ mp4_file +'"'
    process = subprocess.Popen(shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    process.wait()

print(len(merge_list))


