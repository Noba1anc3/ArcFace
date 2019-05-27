# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : EDU_PRODUCT
# @Time    : 19-4-10 上午10:10
# @Author  : kongshuchen
# @FileName: face_Solver.py
# @Software: PyCharm
import cv2
import face_detector.config
import torch
import torch.backends.cudnn as cudnn
from face_detector.layers.functions.prior_box import PriorBox
from face_detector.utils.nms_wrapper import nms
from face_detector.models.faceboxes import FaceBoxes
from face_detector.utils.box_utils import decode
from face_detector.data.config import cfg
import numpy as np

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
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

class Solver(object):

    def __init__(self, config):
        self.trained_model = config.trained_model
        self.cpu = config.cpu
        self.confidence_threshold = config.confidence_threshold
        self.top_k = config.top_k
        self.nms_threshold = config.nms_threshold
        self.keep_top_k = config.keep_top_k

        # load Model
        torch.set_grad_enabled(False)
        # net and model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
        self.net = load_model(self.net, self.trained_model, self.cpu)
        self.net.eval()
        print('Finished loading model!')
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.cpu else "cuda")
        self.net = self.net.to(self.device)


    def response(self, img):
        resize = 1
        im_height, im_width, _ = img.shape
        img = np.float32(img)
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

        result = {"result": dets.tolist()}

        return result




if __name__ == '__main__':
    image_path = "/home/videt/Downloads/WIDER_train/images/0--Parade/0_Parade_marchingband_1_5.jpg"
    image = cv2.imread(image_path)
    solver = Solver(config)
    result = solver.response(image)
    print(result)