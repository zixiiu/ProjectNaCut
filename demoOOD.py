from video_processor.videoSession import videoSession

if __name__ == '__main__':
    a = videoSession("./testVideo/smaller.mp4")
    a.start()
    while True:
        ret = a.nextFrame()
        if ret is None:
            break
        print(ret)