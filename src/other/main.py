# -*- coding: utf-8 -*-

import os 
import sys

import cv2
import numpy as np

from utils.detector import Detector
from utils.visualize import plot_points, plot_slots

HEIGHT_RESIZE = 512
WIDTH_RESIZE  = 512
 

def main():
    root = '../models/'
    filepath = 'parkingslot_fp16.trt'
    img_path = "../imgs/0034.jpg"

    filepath = os.path.join(root, filepath)
    detector = Detector(filepath,(HEIGHT_RESIZE ,WIDTH_RESIZE))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (WIDTH_RESIZE, HEIGHT_RESIZE))

    import time
    s = time.time()
    for i in range(200):
        pred_points, slots = detector.inference(img)
        #print(slots)
    e = time.time()
    print("mean:", (e-s)/200)
    print("FPS:", 200/(e-s))

    plot_points(img, pred_points)
    cv2.imwrite("../imgs/points.jpg", img)    
    plot_slots(img, pred_points, slots)
    cv2.imwrite("../imgs/slots.jpg", img)


if __name__ == '__main__':
    main()

