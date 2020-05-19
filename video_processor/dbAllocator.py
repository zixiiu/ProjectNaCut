from video_processor.ORMModel import *
#from video_processor.videoSession import videoSession
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import cv2 as cv
import tqdm

class Allocator(object):
    def __init__(self, db_path, drive):
        engine = create_engine(db_path)
        sm = sessionmaker()
        sm.configure(bind=engine)
        self.session = sm()
        self.drive = drive

        self.thisCut = None
        self.thisPersonDict = {}

    def getVideoById(self, id):
        return self.session.query(Video).get(id)

    def getFacebyID(self, id):
        return self.session.query(Face).get(id)

    def queryPersonByCut(self, cut_id):
        return self.session.query(Person).filter(Person.cut_id == cut_id).all()

    def queryPersonByKeyfaceId(self, keyface_id):
        ret = self.session.query(Person).filter(Person.keyface_id == keyface_id).all()
        return None if len(ret) != 1 else ret[0]

    def queryAllPerson(self):
        return self.session.query(Person).all()

    def queryFaceByPerson(self, person):
        res = []
        for pif in person.person_in_frame:
            for f in self.session.query(Face).filter(Face.personInFrame_id == pif.id).all():
                res.append(f)
        return res

    def getTotalVideo(self):
        return len(self.session.query(Video).all())

    def getTotalFrame(self):
        total_frame = 0
        for thisVideo in self.session.query(Video).all():
            if thisVideo.complete:
                continue
            v = cv.VideoCapture(self.getFullPath(thisVideo))
            total_frame += int(v.get(cv.CAP_PROP_FRAME_COUNT))
            v.release()
        return total_frame

    def getFullPath(self, thisVideo):
        return os.path.join(self.drive, os.sep, thisVideo.path)

    def getFirstUnProcessed(self):
        for thisVideo in self.session.query(Video).all():
            if thisVideo.complete == False:
                return thisVideo
        return False


    def cleanVid(self, thisVideo):
        session = self.session
        session.query(Cut).filter(Cut.video_id == thisVideo.id).delete()
        session.query(Face).filter(Face.video_id == thisVideo.id).delete()
        session.query(Frame).filter(Frame.video_id == thisVideo.id).delete()
        session.query(Person).filter(Person.video_id == thisVideo.id).delete()
        session.query(PersonInFrame).filter(PersonInFrame.video_id == thisVideo.id).delete()
        session.commit()

    def newVideo(self):
        self.thisCut = None
        self.thisPersonDict = {}

    def videoComplete(self, thisVideo):
        thisVideo.complete = True
        self.session.commit()

    def writeFace(self, x1, y1, x2, y2, PIFID):
        thisPIF = self.session.query(PersonInFrame).get(PIFID)
        # Face(x1=faceDict['x1'], y1=faceDict['y1'], x2=faceDict['x2'], y2=faceDict['y2'],
        #                         x1_abs=person['x1'] + faceDict['x1'],
        #                         x2_abs=person['x1'] + faceDict['x2'], y1_abs=person['y1'] + faceDict['y1'],
        #                         y2_abs=person['y1'] + faceDict['y2'],
        #                         personInFrame=thisPIF, video=thisVideo, at_frame=currFrame, frame=thisFrame)
        thisFace = Face(x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        x1_abs= thisPIF.x1 + x1, x2_abs= thisPIF.x1 + x2,
                        y1_abs=thisPIF.y1 + y1, y2_abs= thisPIF.y1 + y2,
                        at_frame = thisPIF.at_frame, frame= thisPIF.frame,
                        personInFrame=thisPIF, video=thisPIF.video
                        )
        self.session.add(thisFace)
        self.session.commit()


    def write(self, ret, thisVideo):
        session = self.session
        currFrame = ret['frame_no']
        if ret['is_cut'] or self.thisCut is None:  # new cut!
            self.thisCut = Cut(local_id=ret['cut_id'], video=thisVideo, start_frame_no=currFrame)
            self.thisPersonDict = {}
            session.add(self.thisCut)
            session.commit()
        self.thisCut.end_frame_no = currFrame
        thisFrame = Frame(local_frame_no=currFrame, video=thisVideo, cut=self.thisCut)
        session.add(thisFrame)

        for person in ret['person']:
            track_id = person['trackId']
            if track_id not in self.thisPersonDict.keys():
                self.thisPersonDict[track_id] = Person(start_frame_no=currFrame, track_id=track_id, cut=self.thisCut,
                                                  video=thisVideo)
                session.add(self.thisPersonDict[track_id])

            thisPIF = PersonInFrame(track_id=track_id, at_frame=currFrame, x1=person['x1'], y1=person['y1'],
                                    x2=person['x2'], y2=person['y2'],
                                    person=self.thisPersonDict[track_id], video=thisVideo, frame=thisFrame)
            session.add(thisPIF)
            self.thisPersonDict[track_id].end_frame_no = currFrame
            if person['face']:
                faceDict = person['face']
                thisFace = Face(x1=faceDict['x1'], y1=faceDict['y1'], x2=faceDict['x2'], y2=faceDict['y2'],
                                x1_abs=person['x1'] + faceDict['x1'],
                                x2_abs=person['x1'] + faceDict['x2'], y1_abs=person['y1'] + faceDict['y1'],
                                y2_abs=person['y1'] + faceDict['y2'],
                                personInFrame=thisPIF, video=thisVideo, at_frame=currFrame, frame=thisFrame)
                session.add(thisFace)

                thisPIF.face = thisFace
        session.commit()
