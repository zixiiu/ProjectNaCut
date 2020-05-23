import requests
from video_processor import dbAllocator
from video_processor.ORMModel import *
import bs4
import json
import tqdm

al = dbAllocator.Allocator('sqlite:///../testv-complete.sqlite', '/')

dict = {}
for v in tqdm.tqdm(al.session.query(Video).all()):
	av = v.avid
	r = requests.get('https://www.bilibili.com/video/av'+ str(av))
	html = bs4.BeautifulSoup(r.text)
	t = html.title.text
	dict[av] = t

with open('videoTitle.json', 'w', encoding='utf-8') as fp:
	json.dump(dict, fp)