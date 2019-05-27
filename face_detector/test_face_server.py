# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : FaceBoxes.PyTorch
# @Time    : 19-4-10 下午2:13
# @Author  : kongshuchen
# @FileName: test_face_server.py
# @Software: PyCharm
import argparse
import datetime
import math
import cv2
import math
import numpy as np
import time
import datetime
import yaml
import base64
import json
import os
import requests
import threading
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
# Ensure settings are read
# from django.core.wsgi import get_wsgi_application
# application = get_wsgi_application()


import logging
import multiprocessing
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')

class startRead(threading.Thread):
    """docstring for startRead"""

    def __init__(self,RTSPDIR):
        self.RTSPDIR = RTSPDIR
        self.cap = cv2.VideoCapture(RTSPDIR)
        threading.Thread.__init__(self)
        self.ret=0
        self.frame=0
        self.num=0
        self.stop = False

    def stoprun(self):
        self.stop=True

    def run(self):
        self.stop = False
        while(True):
            if self.stop:
                print('stop read')
                break
            try:
                # logging.critical(RTSPDIR)
                self.ret, self.frame = self.cap.read()
                if(self.num>=50 and not self.ret):
                    # logging.critical(self.num)
                    self.num=0
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.RTSPDIR)
                if(not self.ret):
                    self.num+=1
                    # logging.critical(self.num)
                    continue
                else:
                    self.num=0
            except Exception as e:
                logging.critical(e)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.RTSPDIR)



class Detection():

    def __init__(self, videopath):
        self.videopath = videopath

    def callDetect(self, Input, IP):
        # logging.critical(IP)
        r = requests.post(IP, data=Input)
        return json.loads(r.text)

    def saveResult(self,frame,frame_time):
        folder = datetime.datetime.now().strftime("%Y-%m-%d")
        save_path = './disp/'+'/'+folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+'/'+frame_time+'.jpg',frame)

    def Operation(self):
        camera = cv2.VideoCapture(self.videopath)
        grabbed, frame = camera.read()
        # loop over the frames of the video
        t1 = startRead(self.videopath)
        t1.start()
        start = time.time()
        sumnum = 0
        target_image = None
        target_num = 0
        target_time = 0

        while True:
            try:
                if not t1.ret:
                    continue
                frame = t1.frame.copy()
                temp_time = datetime.datetime.now().strftime("%H:%M:%S")
                try:
                    w, h, c = frame.shape
                    # ratio = 416.0 / min(w, h)
                    # frame = cv2.resize(frame, None, None, ratio, ratio, interpolation=cv2.INTER_CUBIC)
                    byimage = base64.b64encode(cv2.imencode(".png", frame.copy())[1].tostring())
                    ans = self.callDetect(byimage, "http://10.5.5.91:10100/face_detect")
                    # logging.critical(ans)
                    temp_class = 0
                    print(ans)

                    for subans in ans["result"]:
                        cv2.rectangle(frame, (int(subans[0]), int(subans[1])), (int(subans[2]), int(subans[3])), (0, 0, 255), 3)
                    target_image = frame.copy()
                    self.saveResult(target_image, temp_time)

                    # cv2.imshow('Face detect Demo', frame.copy())
                    # cv2.waitKey(0)
                    # key = cv2.waitKey(33)
                    # if key == 27:  # esc
                    #     return
                except:
                    pass
                nowtime = datetime.datetime.now().strftime("%H:%M")
                sumnum += 1
                if (time.time() - start) > 1.0:
                    logging.critical(sumnum)
                    sumnum = 0
                    start = time.time()
            except:
                t1.stoprun()
                break



if __name__ == '__main__':
    detection = Detection("./test1.mp4")
    detection.Operation()