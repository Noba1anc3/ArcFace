# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : EDU_PRODUCT
# @Time    : 19-4-10 上午10:10
# @Author  : kongshuchen
# @FileName: face_Solver.py
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

parser = argparse.ArgumentParser(description='FaceBoxes')

# parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes_epoch_200.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes_epoch_270.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='/media/videt/Data1/education/2019-04-27', type=str, help='dataset')
# parser.add_argument('--dataset', default='/media/videt/Data1/PycharmProjects/InsightFace_Pytorch/test_data/', type=str, help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.01, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


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


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    print(device)
    net = net.to(device)

    landmark_predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")


    # save file
    # if not os.path.exists(args.save_folder):
    #     os.makedirs(args.save_folder)
    # fw = open(os.path.join(args.save_folder, args.dataset + '_dets.txt'), 'w')

    # testing dataset
    # testset_folder = os.path.join(args.dataset, 'images/')
    testset_folder = "/media/videt/Data/PycharmProjects/FaceBoxes.PyTorch/data/WIDER_FACE/images/1--Handshaking/"
    # testset_folder = "/media/videt/Data1/education/2019-04-27/"
    test_dataset = os.listdir(testset_folder)
    num_images = len(test_dataset)

    # testing scale
    resize = 1

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        print(image_path)
        image = (cv2.imread(image_path, cv2.IMREAD_COLOR))
        img = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        print(img_name, img.shape)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        out = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(device)
        loc, conf, _ = out   # conf: (num_boxes, num_classes)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores, labels = np.max(conf.data.cpu().numpy(), axis=1), np.argmax(conf.data.cpu().numpy(), axis=1)
        #
        # ignore low score
        t = np.where(scores[np.where(labels != 0)] > args.confidence_threshold)[0]
        inds = [np.where(labels != 0)[0][ele] for ele in t]

        # scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        # inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        labels = labels[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        print(dets.shape)
        # save dets
        if args.dataset == "FDDB":

            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                cv2.imshow(img)
                cv2.rectangle(img, pt1=(xmin, ymin), pt2=(xmax, ymax))
                cv2.waitKey(0)
        else:
            landmarks = dlib.full_object_detections()
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]

                ymin += 0.2 * (ymax - ymin + 1)
                score = dets[k, 4]

                face = dlib.rectangle(left=int(xmin), top=int(ymin), right=int(xmax), bottom=int(ymax))
                shape = landmark_predictor(image, face)
                landmarks.append(shape)
                for i, pt in enumerate(shape.parts()):
                    cv2.circle(image, (pt.x, pt.y), 1, (0, 255, 0), -1, 8)
                    # cv2.putText(image, str(i), (pt.x, pt.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                cv2.rectangle(img=image, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(0, 0, 255), thickness=2)

            if len(landmarks) != 0:
                aligned_faces = dlib.get_face_chips(image, landmarks)
                image_cnt = 0
                for aligned_face in aligned_faces:
                    image_cnt += 1
                    cv_rgb_aligned_face = np.array(aligned_face).astype(np.uint8)  # 先转换为numpy数组
                    # cv_bgr_ialigned_face = cv2.cvtColor(cv_rgb_aligned_face, cv2.COLOR_RGB2BGR)  # opencv下颜色空间为bgr，所以从rgb转换为bgr
                    cv2.imshow('%s' % (image_cnt), cv_rgb_aligned_face)

            _t['misc'].toc()

            # image = cv2.resize(image, None, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
