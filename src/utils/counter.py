import time

class Counter:
    def __init__(self):
        self.sum = 0
        self.count = 0  
        self.ave = 0      
        self.s = 0
        self.e = 0

    def start(self):
        self.s = time.time()

    def end(self):
        self.e = time.time()
        self.sum += self.e - self.s
        self.count += 1
        self.ave = self.sum / self.count
        print("time: ", self.ave * 1000, "ms")
        print("FPS: ", 1 / self.ave)
