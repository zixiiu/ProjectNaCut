def yoloDetectProcess(stopped, inQueue, outQueue, lastFrameNumber):
    yolo = YOLO()
    while True:
        if stopped:
            break

        if not inQueue.empty():
            franeNo, frame = inQueue.get()
            boxs = yolo.detect_image(frame)

            while lastFrameNumber.value != franeNo - 1:
                time.sleep(0.01)

            outQueue.put((franeNo, frame, boxs))
            lastFrameNumber.value = franeNo

        else:
            time.sleep(0.1)
