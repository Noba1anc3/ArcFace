# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : EDU_PRODUCT
# @Time    : 19-4-10 上午10:10
# @Author  : kongshuchen
# @FileName: face_Solver.py
# @Software: PyCharm

cfg = {
    'name': 'FaceBoxes',
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'min_dim': 1024,
    'steps': [32, 64, 128],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'aspect_ratios': [[1], [1], [1]],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}
