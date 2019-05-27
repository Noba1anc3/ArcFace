# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : EDU_PRODUCT
# @Time    : 19-4-30 下午8:04
# @Author  : kongshuchen
# @FileName: face_detector_inference.py
# @Software: PyCharm

from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from face_detector.data.config import cfg
from face_detector.layers.functions.prior_box import PriorBox
from face_detector.utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from face_detector.models.faceboxes import FaceBoxes
from face_detector.utils.box_utils import decode
from face_detector.utils.timer import Timer
import dlib
import face_detector.config as config

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #print('Missing keys:{}'.format(len(missing_keys)))
    #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('******Loading Face Detection Model from {}******'.format(pretrained_path))
    print('')
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class FaceDetectorInference(object):
    def __init__(self, config, landmark_flag, aligned_flag):
        self.trained_model = config.trained_model
        self.cpu = config.cpu
        self.confidence_threshold = config.confidence_threshold
        self.top_k = config.top_k
        self.nms_threshold = config.nms_threshold
        self.keep_top_k = config.keep_top_k
        self.landmark_flag = landmark_flag
        self.aligned_flag = aligned_flag

        # load Model
        torch.set_grad_enabled(False)
        # net and model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
        self.net = load_model(self.net, self.trained_model, self.cpu)
        self.net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.cpu else "cuda")
        self.net = self.net.to(self.device)
        self.landmark_predictor = dlib.shape_predictor("../face_detector/weights/shape_predictor_68_face_landmarks.dat")

        print('***********************Detection and Recognition Model Loaded***********************')
        print('')

    def infer(self, image):
        """

        :param img: the input image
        :return: a list of face image, landmarks
        """
        #cv2.imwrite('.\image',image)
        img = np.float32(image)
        resize = 1
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        out = self.net(img)  # forward pass
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(self.device)
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, self.nms_threshold, force_cpu=self.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]


        landmarks = dlib.full_object_detections()

        faces_list = []
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            # ymin += 0.2 * (ymax - ymin + 1)
            # score = dets[k, 4]
            if not self.aligned_flag:
                faces_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax)])

                # cv2.imshow("image", image[int(ymin):int(ymax), int(xmin):int(xmax)])
                # print(int(ymin),int(ymax), int(xmin),int(xmax))

            if self.landmark_flag:
                face = dlib.rectangle(left=int(xmin), top=int(ymin), right=int(xmax), bottom=int(ymax))
                shape = self.landmark_predictor(image, face)
                landmarks.append(shape)
            #     for i, pt in enumerate(shape.parts()):
            #         cv2.circle(image, (pt.x, pt.y), 1, (0, 255, 0), -1, 8)
            #
            # cv2.rectangle(img=image, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(0, 0, 255),
            #               thickness=2)

        if len(landmarks) != 0 and self.aligned_flag:
            aligned_faces = dlib.get_face_chips(image, landmarks)
            image_cnt = 0
            for aligned_face in aligned_faces:
                image_cnt += 1
                cv_rgb_aligned_face = np.array(aligned_face).astype(np.uint8)  # 先转换为numpy数组
                faces_list.append(cv_rgb_aligned_face)
            #     cv2.imshow('%s' % (image_cnt), cv_rgb_aligned_face)
            # cv2.waitKey(0)

        return dets[:, :-1], faces_list, landmarks


if __name__ == '__main__':

    face_detector = FaceDetectorInference(config=config, landmark_flag=True, aligned_flag=True)
    img = cv2.imread("1.png")
    boxes, face_list, landmarks = face_detector.infer(img)

    print(boxes.shape)

    for face in face_list:
        face.save("1.png", 'png')

    for landmark in landmarks:
        print(landmark.parts())