# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : EDU_PRODUCT
# @Time    : 19-4-10 上午10:10
# @Author  : kongshuchen
# @FileName: face_Solver.py
# @Software: PyCharm
from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data
# from data.wider_voc import AnnotationTransform, VOCDetection, detection_collate
from face_detector.data.safe_product import AnnotationTransform, VOCDetection, detection_collate
from face_detector.data.config import  cfg
from face_detector.data.data_augment import preproc
from face_detector.layers.modules import multibox_loss
from face_detector.layers.functions.prior_box import PriorBox
import time
import math
from face_detector.models.faceboxes import FaceBoxes
import tensorboardX


parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--training_dataset', default='/media/videt/Data/PycharmProjects/FaceBoxes.PyTorch/data/WIDER_FACE/', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=6, type=int, help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default="./weights/FaceBoxes_epoch_170.pth", help='resume net for retraining')
parser.add_argument('--resume_epoch', default=140, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--output_dir', default='./output/')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

img_dim = 1024
rgb_means = (104, 117, 123) #bgr order
num_classes = 3
batch_size = args.batch_size
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum
gpu_train = cfg['gpu_train']

net = FaceBoxes('train', img_dim, num_classes)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

device = torch.device('cuda:0' if gpu_train else 'cpu')
cudnn.benchmark = True
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = multibox_loss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = VOCDetection(args.training_dataset, preproc(img_dim, rgb_means), AnnotationTransform())
    summary_writer = tensorboardX.SummaryWriter(log_dir=args.output_dir)

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    max_iter = args.max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder + 'FaceBoxes_epoch_' + repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)


        batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers,
                                              collate_fn=detection_collate))

        # load train data
        images, targets = next(batch_iterator)
        if gpu_train:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) +
              '|| Totel iter ' + repr(iteration) + ' || L: %.4f C: %.4f||' % (cfg['loc_weight']*loss_l.item(), loss_c.item()) +
              'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
        summary_writer.add_scalar("losses/total_loss", loss.item(), global_step=iteration)
        summary_writer.add_scalar("losses/loss_l", loss_l.item(), global_step=iteration)
        summary_writer.add_scalar("losses/loss_c", loss_c.item(), global_step=iteration)

    torch.save(net.state_dict(), args.save_folder + 'Final_FaceBoxes.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 0:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
if __name__ == '__main__':
    train()
