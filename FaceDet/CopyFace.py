import os
import shutil
import json
import tqdm

with open('DNNdetection.json') as json_file:
    det = json.load(json_file)

src = './face'
tar = './nhqDNN'

for id in tqdm.tqdm(det):
    cls = det[id][0]
    sco = int(round(det[id][1]*100,0))
    if cls == 0:
        shutil.copyfile(os.path.join(src, str(id)+'.jpg'), os.path.join(tar, str(sco)+'_'+str(id)+'.jpg'))
