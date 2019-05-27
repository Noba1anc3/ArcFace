# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : EDU_PRODUCT
# @Time    : 19-4-10 上午10:10
# @Author  : kongshuchen
# @FileName: face_Solver.py
# @Software: PyCharm
HOST = '0.0.0.0'
PORT = 10100

trained_model = "../face_detector/weights/FaceBoxes.pth"
cpu = False
confidence_threshold = 0.5
top_k = 5000
nms_threshold = 0.1
keep_top_k = 750