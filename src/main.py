# -*- coding: utf-8 -*-

import os 
import sys

import cv2
import numpy as np
import time

from utils.detector import Detector
from utils.visualize import plot_points, plot_slots
from utils.counter import Counter

HEIGHT_RESIZE = 512
WIDTH_RESIZE  = 512
VIDEO = "/home/nvidia/robot/video/videos/ps.avi"
VIS = True
COUNT = True

def main():
    root = '../models/'
    filepath = 'parkingslot_fp16.trt'
    filepath = os.path.join(root, filepath)
    detector = Detector(filepath,(HEIGHT_RESIZE ,WIDTH_RESIZE))
    counter = Counter() 

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("camera not open")
        exit()

    while True:
        counter.start()
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH_RESIZE, HEIGHT_RESIZE))
        pred_points, slots = detector.inference(img)

        if VIS:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #plot_points(img, pred_points)
            plot_slots(img, pred_points, slots)
            cv2.imshow("parking_slot", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if COUNT:
            counter.end()

if __name__ == '__main__':
    main()

