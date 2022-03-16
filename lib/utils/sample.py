# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.resnet_pre import resnet_pre
from model.faster_rcnn.resnet_cur import resnet_cur
from datasets.tasks import *
from sample_aid import *

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--dataset_un', dest='dataset_un',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res50.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--task', dest='task',
                        help='number of classes to train',
                        default="19-1", type=str)
    parser.add_argument('--step', dest='step',
                        help='which step to train',
                        default=1, type=int)
    parser.add_argument('--num_classes_old', dest='num_classes_old',
                        help='number of classes to train',
                        default=20, type=int)
    parser.add_argument('--num_classes_new', dest='num_classes_new',
                        help='number of classes to train',
                        default=2, type=int)
    parser.add_argument('--num_classes', dest='num_classes',
                        help='number of classes to train',
                        default=20, type=int)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="/home/zhangc/PycharmProjects/faster-rcnn-incremental-non-overlaps.pytorch/models_pre",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession_pre', dest='checksession_pre',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch_pre', dest='checkepoch_pre',
                        help='checkepoch to load network',
                        default=20, type=int)
    parser.add_argument('--checkpoint_pre', dest='checkpoint_pre',
                        help='checkpoint to load network',
                        default=9463, type=int)
    parser.add_argument('--checksession_cur', dest='checksession_cur',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch_cur', dest='checkepoch_cur',
                        help='checkepoch to load network',
                        default=20, type=int)
    parser.add_argument('--checkpoint_cur', dest='checkpoint_cur',
                        help='checkpoint to load network',
                        default=147, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--open', dest='open',
                        help='open',
                        default=False, type=bool)
    parser.add_argument('--pseudo_gt_boxes_pre_path', dest='pseudo_gt_boxes_pre_path',
                        default="",
                        type=str)
    parser.add_argument('--pseudo_gt_boxes_cur_path', dest='pseudo_gt_boxes_cur_path',
                        default="",
                        type=str)
    parser.add_argument('--pseudo_gt_boxes_num_path', dest='pseudo_gt_boxes_num_path',
                        default="",
                        type=str)
    parser.add_argument('--img_id_path', dest='img_id_path',
                        default="",
                        type=str)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc" and args.dataset_un == "coco":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        args.imdb_name_un = "coco_2014_train"  # +coco_2014_valminusminival
        args.imdbval_name_un = "coco_2014_minival"
        args.set_cfgs_un = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "./cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "./cfgs/{}.yml".format(
        args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False

    labels, labels_old = get_task_labels(args.dataset, args.task, args.step)
    imdb, _, _, _ = combined_roidb(args.imdb_name)
    labels.insert(0, 0)
    labels_old.insert(0, 0)
    classes_old = [imdb.classes[i] for i in labels_old]
    classes = [imdb.classes[i] for i in labels]
    # unlabeled dataset coco
    imdb_un, roidb_un, ratio_list_un, ratio_index_un = combined_roidb(args.imdb_name_un)

    imdb_un.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb_un)))

    input_dir = args.load_dir + "/" + args.net + "/" + "pascal_voc"  # args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name_pre = os.path.join(input_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession_pre, args.checkepoch_pre,
                                                                      args.checkpoint_pre))
    load_name_cur = os.path.join(input_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession_cur, args.checkepoch_cur,
                                                                              args.checkpoint_cur))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN_pre = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN_pre_un = resnet_pre(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN_pre = resnet_pre(classes_old, 50, pretrained=False, class_agnostic=args.class_agnostic)
        fasterRCNN_cur = resnet_cur(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN_pre_un = resnet_pre(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN_pre.create_architecture()
    fasterRCNN_cur.create_architecture()

    print("load checkpoint %s" % (load_name_pre))
    checkpoint_pre = torch.load(load_name_pre)
    fasterRCNN_pre.load_state_dict(checkpoint_pre['model'])

    print("load checkpoint %s" % (load_name_cur))
    checkpoint_cur = torch.load(load_name_cur)
    fasterRCNN_cur.load_state_dict(checkpoint_cur['model'])

    if 'pooling_mode' in checkpoint_pre.keys():
        cfg.POOLING_MODE = checkpoint_pre['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    img_id = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        img_id = img_id.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    img_id = Variable(img_id)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN_pre.cuda()
        fasterRCNN_cur.cuda()

    start = time.time()
    num_images = len(imdb_un.image_index)
    dataset = roibatchLoader(roidb_un, ratio_list_un, ratio_index_un, 1, imdb_un.num_classes, training=True,
                             normalize=False)  # imdb.num_classes
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)

    sample_pseudo_gt_boxes = sample_pseudo_gt_boxes(args)
    per_class_threshold_pre = []
    per_class_threshold_cur = []
    img_selected = []
    gt_boxes_pre_selected = []
    gt_boxes_cur_selected = []
    pseudo_gt_boxes_num = []
    train_size = len(roidb_un)
    for i in range(train_size):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            # img_id.resize_(data[4].size()).copy_(data[4])
            img_id = data[4]

        fasterRCNN_pre.eval()
        fasterRCNN_cur.eval()
        gt_boxes_pre, num_boxes_pre = sample_pseudo_gt_boxes.__call__(fasterRCNN_pre, im_data, im_info, gt_boxes,
                                                                      num_boxes, args.num_classes_old,
                                                                      per_class_threshold_pre)
        gt_boxes_cur, num_boxes_cur = sample_pseudo_gt_boxes.__call__(fasterRCNN_cur, im_data, im_info, gt_boxes,
                                                                      num_boxes, args.num_classes_new,
                                                                      per_class_threshold_cur)
        n_pre = num_boxes_pre.item()
        n_cur = num_boxes_cur.item()
        if 0 < n_pre + n_cur <= 20:
            img_selected.append(img_id)
            gt_boxes_pre_selected.append(gt_boxes_pre)
            print(gt_boxes_pre)
            gt_boxes_cur_selected.append(gt_boxes_cur)
            print(gt_boxes_cur)
            pseudo_gt_boxes_num.append(torch.Tensor([n_pre, n_cur]).cuda())
    np.save(args.img_id_path, img_selected)
    np.save(args.pseudo_gt_boxes_pre_path, gt_boxes_pre_selected)
    np.save(args.pseudo_gt_boxes_cur_path, gt_boxes_cur_selected)
    np.save(args.pseudo_gt_boxes_num_path, pseudo_gt_boxes_num)
    end = time.time()
    print("sample time: %0.4fs" % (end - start))
