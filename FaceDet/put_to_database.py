import os
import json
import tqdm
from video_processor import dbAllocator

with open('DNNdetection.json') as json_file:
	det = json.load(json_file)

src = './face'
tar = './nhqDNN'

rela = {0: 'nvhouqi',
        1: 'nanhouqi',
        2: 'hanhan',
        3: 'shushu',
        4: 'qunyan',
        5: 'laoban',
        6: 'baindaoBF',
        7: 'xinyue',
        8: 'biandao',
        9: 'py',
        10: 'xiaoshuaige',
        11: 'other',
        -1: 'unknown'}

al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')

for id in tqdm.tqdm(det):
    cls = det[id][0]
    sco = int(round(det[id][1] * 1000, 0))
    per = al.queryPersonByKeyfaceId(id)
    per.name_mark = rela[cls]
    per.name_conf = sco

al.session.commit()