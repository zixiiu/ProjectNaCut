from video_processor.videoSession import videoSession

if __name__ == '__main__':
    a = videoSession("./testVideo/604_0_new.mp4", visualize=True)
    a.start()
    while True:
        ret = a.nextFrame()
        if ret is None:
            break
        print(ret)