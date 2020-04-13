from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from video_processor.ORMModel import *
from video_processor.videoSession import videoSession
from video_processor.dbAllocator import Allocator
import tqdm
import os

if __name__ == '__main__':

    al = Allocator('sqlite:///testv.sqlite','/media/seb101-user/New Volume/')
    total_frame = al.getTotalFrame()
    print(total_frame)
    vidPbar = tqdm.tqdm(total = al.getTotalVideo(), desc = 'Video:', position= 0)
    framePbar = tqdm.tqdm(total = total_frame, desc = 'Total frame:', position= 1)


    while True:
        thisVideo = al.getFirstUnProcessed()
        if thisVideo is False:
            print('all jobs done')
            break
        vidPbar.update(1)
        al.newVideo()
        al.cleanVid(thisVideo)
        thisVideoPath = al.getFullPath(thisVideo)
        a = videoSession(thisVideoPath, visualize=False, cnn=False)
        a.start()

        thisVidPbar = tqdm.tqdm(total = a.video_capture.getFrameCount(), desc="this Video", position= 2)
        while True:
            thisVidPbar.update(1)
            framePbar.update(1)
            ret = a.nextFrame()
            if ret is None:
                print('video complete')
                al.videoComplete(thisVideo)
                a.release()
                break
            #{'frame_no': 388, 'is_cut': False, 'cut_id': 3, 'person': [{'trackId': 3, 'x1': 314, 'y1': 56, 'x2': 1656, 'y2': 1039, 'face': {'x1': 461, 'x2': 846, 'y1': 162, 'y2': 547}}]}
            #print(ret)
            al.write(ret,thisVideo)





