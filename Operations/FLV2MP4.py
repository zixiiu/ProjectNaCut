import tqdm
import os
import subprocess

base_dir = "D:\Testv\VideoOnly"

toDelete = []
for video in tqdm.tqdm(os.listdir(base_dir)):
    if video.endswith('.flv'):#need convert
        shell = "ffmpeg -y -hwaccel cuvid -hwaccel_device 0 -i " + os.path.join(base_dir, video) + " -c:v h264_nvenc -crf 19 -strict experimental " + os.path.join(base_dir, video.split('.')[0] + ".mp4")
        #print(shell)
        process = subprocess.Popen(shell, stdout=None, stderr=None, shell=True)
        process.wait()
        toDelete.append(os.path.join(base_dir, video))
