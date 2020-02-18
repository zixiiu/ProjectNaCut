import face_recognition
import time

def faceDetectProcess(stopped, inQueue, outQueue, lastFrameNumber):

    while True:
        if stopped:
            break

        if not inQueue.empty():
            (grabbed, curr_frame, frame, isCut, detections) = inQueue.get()
            face_abs_list = []
            for det in detections:
                bbox = det.to_tlbr()
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                peopleFrame = frame[y1:y2, x1:x2]
                face_locations = face_recognition.face_locations(peopleFrame, model='cnn')
                for top, right, bottom, left in face_locations:
                    top += y1
                    bottom += y1
                    right += x1
                    left += x1
                    face_abs_list.append((top, right, bottom, left))

            while lastFrameNumber.value != curr_frame - 1:
                time.sleep(0.01)

            outQueue.put((grabbed, curr_frame, frame, isCut, detections, face_abs_list))
            lastFrameNumber.value = curr_frame

        else:
            time.sleep(0.1)
