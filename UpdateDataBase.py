from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, DateTime, String, Integer, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

from video_processor.ORMModel import *
import os


newfile = ~os.path.isfile('./test.sqlite')
engine = create_engine('sqlite:///test.sqlite')

session = sessionmaker()
session.configure(bind=engine)
if newfile:
    Base.metadata.create_all(engine)

s = session()
cvSet = set()
for i in s.query(Video).all():
    cvSet.add(i.cvid)


for root, dirs, files in os.walk("F:\\Testv2\\11336264"):
    for name in files:
        if name.endswith('.mp4'):
            #print(os.path.join(root, name))
            thisCvid = name.split('_')[0]
            thisAvid = root.split("\\")[-1]
            fullpath = os.path.join(root, name)
            childPath = os.path.splitdrive(fullpath)[1]
            if thisCvid not in cvSet:
                s.add(Video(path = childPath, complete= False, avid=thisAvid, cvid=thisCvid))
                s.commit()



