from multiprocessing import Process, Queue
import time

def printQ(q):
    while True:
        print(q.get())
        time.sleep(1)

mpq = Queue()

if __name__ == '__main__':
    for i in range(20):
        mpq.put(i)
    for i in range(3):
        p = Process(target=printQ, args=(mpq,))
        p.start()
        time.sleep(0.33)
