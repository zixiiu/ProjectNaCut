import Util.filevideostream as filevideostream
import os
import pandas as pd
import cv2
from skimage.metrics import structural_similarity
import tqdm

class FrameDiff(object):
    def __init__(self, video_path):
        self.vid = filevideostream.FileVideoStream(video_path).start()
        dirName = os.path.dirname(video_path)
        fileName = os.path.basename(video_path)
        self.targetCsv = os.path.join(dirName, fileName.split(".mp4")[0] + ".csv")
        self.df = None
        self.init_panda()

    def run(self):
        lastFrame=None
        frameNo = 0

        pbar = tqdm.tqdm(total=int(self.vid.getFrameCount()), desc='Cropping...')
        while True:
            curr_frame, frame = self.vid.read()

            if frame is None:
                break

            thisFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frameNo != 0:
                score = structural_similarity(thisFrame, lastFrame)
                self.df.at[frameNo, 'score'] = score
                self.df.at[frameNo, 'Frame'] = frameNo

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            lastFrame = thisFrame
            frameNo += 1
            pbar.update(1)

        self.save_csv()

    def save_csv(self):
        self.df.to_csv(self.targetCsv)

    def init_panda(self):
        self.df = pd.DataFrame(columns=['Frame', 'score'])

if __name__ == "__main__":
    a = FrameDiff("D:\Testv\TESTV官方频道\《值不值得买》2017新年特别节目——你好！我是女后妻。3Q!/《值不值得买》2017新年特别节目——你好！我是女后妻。3Q!_P1_高清 1080P.mp4")
    a.run()