from sqlalchemy import Column, DateTime, String, Integer, Boolean, ForeignKey, func
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Video(Base):
    __tablename__ = 'video'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    avid = Column(Integer)
    cvid = Column(Integer)
    complete = Column(Boolean)

    cut = relationship("Cut", back_populates="video")


class Cut(Base):
    __tablename__ = 'cut'
    id = Column(Integer, primary_key=True)
    local_id = Column(Integer)
    start_frame_no = Column(Integer)
    end_frame_no = Column(Integer)

    video_id = Column(Integer, ForeignKey('video.id'))
    video = relationship(Video, back_populates="cut")

    frame = relationship("Frame", back_populates="cut")

class Frame(Base):
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    local_frame_no = Column(Integer)

    cut_id = Column(Integer, ForeignKey('cut.id'))
    cut = relationship(Cut, back_populates='frame')

    video_id =  Column(Integer, ForeignKey('video.id'))
    video = relationship(Video)

class Person(Base):
    __tablename__ = 'person'
    id = Column(Integer, primary_key=True)
    track_id = Column(Integer)
    start_frame_no = Column(Integer)
    end_frame_no = Column(Integer)
    name_mark = Column(String)

    cut_id = Column(Integer, ForeignKey('cut.id'))
    cut = relationship(Cut)

    video_id =  Column(Integer, ForeignKey('video.id'))
    video = relationship(Video)

    person_in_frame = relationship("PersonInFrame", back_populates="person")

class PersonInFrame(Base):
    __tablename__ = 'person_in_frame'
    id = Column(Integer, primary_key=True)
    track_id = Column(Integer)
    at_frame = Column(Integer)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)

    person_id = Column(Integer, ForeignKey('person.id'))
    person = relationship(Person, back_populates="person_in_frame")

    frame_id = Column(Integer, ForeignKey('frame.id'))
    frame = relationship(Frame)

    face=relationship("Face", uselist = False, back_populates = "personInFrame")

    video_id =  Column(Integer, ForeignKey('video.id'))
    video = relationship(Video)

class Face(Base):
    __tablename__ = 'face'
    id = Column(Integer, primary_key=True)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    at_frame = Column(Integer)
    x1_abs = Column(Integer)
    y1_abs = Column(Integer)
    x2_abs = Column(Integer)
    y2_abs = Column(Integer)

    personInFrame_id = Column(Integer, ForeignKey("person_in_frame.id"))
    personInFrame = relationship(PersonInFrame, back_populates="face")

    frame_id = Column(Integer, ForeignKey('frame.id'))
    frame = relationship(Frame)

    video_id =  Column(Integer, ForeignKey('video.id'))
    video = relationship(Video)

if __name__ == "__main__":
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///test.sqlite')

    from sqlalchemy.orm import sessionmaker
    session = sessionmaker()
    session.configure(bind=engine)
    Base.metadata.create_all(engine)

    s = session()
    v1 = Video(path='12345',complete=False)
    s.add(v1)
    c1 = Cut(local_id=1,video=v1)
    s.add(c1)

    s.commit()
    for i in s.query(Video).all():
        print(i.id)
        print(i.path)