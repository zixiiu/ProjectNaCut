from skimage.measure import compare_ssim
import cv2
import queue


class CutDetector(object):
    def __init__(self, threshold):
        self.lastFrame = None
        self.avg = None
        self.isCut = False
        self.noFrame = 0
        self.threshold = threshold
        self.score = 0
        self.scoreQueue = queue.Queue(maxsize=20)

    def putFrame(self, newFrame):
        newFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
        if self.lastFrame is not None:
            score = compare_ssim(newFrame, self.lastFrame, full=False, gradient=False)
            self.score = score
            if self.avg:
                if self.avg - score > self.threshold:  # is cut!
                    self.avg = None
                    self.lastFrame = newFrame
                    self.scoreQueue.queue.clear()
                    self.noFrame = 0
                    return True
                else:  # not cut! update avg
                    if self.scoreQueue.full():
                        toDeleteScore = self.scoreQueue.get()
                        self.avg = (self.avg * (20) - toDeleteScore) / (19)

                    self.scoreQueue.put(score)
                    self.avg = (self.avg * (self.scoreQueue.qsize() - 1) + score) / (self.scoreQueue.qsize())
                    self.noFrame += 1
                    self.lastFrame = newFrame
                    return False
            else:  # no avg, this cut just started
                self.avg = score
                self.scoreQueue.put(score)
                self.noFrame += 1
                self.lastFrame = newFrame
                return False

        else:  # first Frame
            self.noFrame += 1
            self.lastFrame = newFrame
            return False
