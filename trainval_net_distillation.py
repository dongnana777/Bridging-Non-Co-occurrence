# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.resnet_pre import resnet_pre
from model.faster_rcnn.resnet_cur import resnet_cur
from model.faster_rcnn.resnet_pre_un import resnet_pre_un
from datasets.tasks import *
from util import *
import torch.nn.functional as F

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
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
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
    parser.add_argument('--num_classes_un', dest='num_classes_un',
                        help='number of classes to train',
                        default=80, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--open', dest='open',
                        help='open',
                        default=True, type=bool)
    parser.add_argument('--T', dest='T',
                        help='temperature',
                        default=1, type=int)
    parser.add_argument('--ws', dest='warm_step',
                        help='warm up the training',
                        default=400, type=int)
    parser.add_argument('--pseudo_gt_boxes_pre_path', dest='pseudo_gt_boxes_pre_path',
                        help='directory to save images index',
                        default="./faster-rcnn-incremental-non-overlaps.pytorch/pseudo_gt_boxes_pre_path_.npy",
                        type=str)
    parser.add_argument('--pseudo_gt_boxes_cur_path', dest='pseudo_gt_boxes_cur_path',
                        help='directory to save images index',
                        default="./faster-rcnn-incremental-non-overlaps.pytorch/pseudo_gt_boxes_cur_path_.npy",
                        type=str)
    parser.add_argument('--pseudo_gt_boxes_num_path', dest='pseudo_gt_boxes_num_path',
                        help='directory to save images index',
                        default="./faster-rcnn-incremental-non-overlaps.pytorch/pseudo_gt_boxes_num_path_.npy",
                        type=str)
    parser.add_argument('--img_id_path', dest='img_id_path',
                        help='directory to save images index',
                        default="./faster-rcnn-incremental-non-overlaps.pytorch/img_id_path_.npy",
                        type=str)
    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc" and args.dataset_un == "coco":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        args.imdb_name_un = "coco_2014_train"  # +coco_2014_valminusminival
        args.imdbval_name_un = "coco_2014_minival"
        args.set_cfgs_un = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train"  # +coco_2014_valminusminival
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda

    labels, labels_old = get_task_labels(args.dataset, args.task, args.step)
    imdb, _, _, _ = combined_roidb(args.imdb_name)
    labels_old.insert(0, 0)
    labels.insert(0, 0)
    classes_old = [imdb.classes[i] for i in labels_old]
    classes = [imdb.classes[i] for i in labels]
    # sampling coco as unlabeled data
    labels_un = [5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    labels_old_un = [0]
    img_selected = np.load(args.img_id_path, allow_pickle=True)
    gt_boxes_pre = np.load(args.pseudo_gt_boxes_pre_path, allow_pickle=True)
    gt_boxes_cur = np.load(args.pseudo_gt_boxes_cur_path, allow_pickle=True)
    gt_boxes_num = np.load(args.pseudo_gt_boxes_num_path, allow_pickle=True)
    imdb_un, roidb_un, ratio_list_un, ratio_index_un = combined_roidb(args.imdb_name_un, labels_un, labels_old_un, img_selected, gt_boxes_pre, gt_boxes_cur, gt_boxes_num, choice=True)


    train_size = len(roidb_un)

    print('{:d} roidb entries'.format(len(roidb_un)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb_un, ratio_list_un, ratio_index_un, args.batch_size, args.num_classes_un, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes_pre = torch.FloatTensor(1)
    gt_boxes_cur = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes_pre = gt_boxes_pre.cuda()
        gt_boxes_cur = gt_boxes_cur.cuda()


    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes_pre = Variable(gt_boxes_pre)
    gt_boxes_cur = Variable(gt_boxes_cur)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_pre = vgg16(classes_old, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_pre = resnet_pre(classes_old, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_pre = resnet_pre(classes_old, 50, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_cur = resnet_cur(classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_pre = resnet_pre(classes_old, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    # previous model
    fasterRCNN_pre.create_architecture()
    supervisor_path = "/home/zhangc/PycharmProjects/faster-rcnn-incremental-non-overlaps.pytorch/models_pre/res50/pascal_voc/faster_rcnn_1_20_6003.pth"
    print("load checkpoint %s" % (supervisor_path))
    checkpoint = torch.load(supervisor_path)
    fasterRCNN_pre.load_state_dict(checkpoint['model'])

    # current model
    fasterRCNN_cur.create_architecture()
    supervisor_path = "/home/zhangc/PycharmProjects/faster-rcnn-incremental-non-overlaps.pytorch/models_pre/res50/pascal_voc/faster_rcnn_1_20_6679.pth"
    print("load checkpoint %s" % (supervisor_path))
    checkpoint = torch.load(supervisor_path)
    fasterRCNN_cur.load_state_dict(checkpoint['model'])

    for param in fasterRCNN_pre.parameters():
        param.requires_grad = False
    for param in fasterRCNN_pre.RCNN_base.parameters():
        param.requires_grad = False

    for param in fasterRCNN_cur.parameters():
        param.requires_grad = False
    for param in fasterRCNN_cur.RCNN_base.parameters():
        param.requires_grad = False

    fasterRCNN.create_architecture()

    checkpoint_pre = torch.load(supervisor_path)
    checkpoint_cur = torch.load(supervisor_path)
    pretrained_dict = fasterRCNN.state_dict()
    for [k, v], [k1, v1], [k2, v2] in zip(fasterRCNN.state_dict().items(), checkpoint_pre['model'].items(), checkpoint_cur['model'].items()):
        if v.size() == v1.size():
            pretrained_dict[k] = v1
        else:
            n = len(list(v.size()))
            if n == 1:
                m = int(v1.size(0))
                pretrained_dict[k][:m] = v1
                pretrained_dict[k][m:] = v2[-(pretrained_dict[k].size(0)-v1.size(0)):]
            else:
                m = int(v1.size(0))
                pretrained_dict[k][:m, :] = v1
                pretrained_dict[k][m:, :] = v2[-(pretrained_dict[k].size(0)-v1.size(0)):, :]
    fasterRCNN.load_state_dict(pretrained_dict)

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()
        fasterRCNN_pre.cuda()
        fasterRCNN_cur.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
        fasterRCNN_pre = nn.DataParallel(fasterRCNN_pre)
        fasterRCNN_cur = nn.DataParallel(fasterRCNN_cur)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")
    mask_gen = mask_gen(args)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        loss_dis_fm_temp = 0
        loss_dis_det_temp = 0
        loss_roi_fm_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes_pre.resize_(data[2].size()).copy_(data[2])
                gt_boxes_cur.resize_(data[3].size()).copy_(data[3])
                num_boxes.resize_(data[4].size()).copy_(data[4])
            num_boxes_pre = num_boxes[:, :1].squeeze(0)
            num_boxes_cur = num_boxes[:, 1:].squeeze(0)
            gt_boxes_pre = gt_boxes_pre.squeeze(0)
            gt_boxes_cur = gt_boxes_cur.squeeze(0)

            n_pre = num_boxes_pre.item()
            n_cur = num_boxes_cur.item()
            if 0 < n_cur + n_pre <= 20:
                gt_boxes_joint = torch.zeros([1, 20, 5], dtype=torch.float).cuda()
                gt_boxes_joint[:, :n_pre, :] = gt_boxes_pre[:, :n_pre, :]
                gt_boxes_joint[:, n_pre:(n_pre + n_cur), :] = gt_boxes_cur[:, :n_cur, :]
                gt_boxes_joint[:, n_pre:(n_pre + n_cur), 4:] = gt_boxes_joint[:, n_pre:(n_pre + n_cur), 4:] + float(
                    args.num_classes_old - 1)
                num_boxes_joint = torch.IntTensor([n_cur + n_pre])

                ######previous model
                fasterRCNN_pre.train()
                rois_pre, cls_score_pre, bbox_pred_pre, \
                rpn_loss_cls_pre, rpn_loss_box_pre, \
                RCNN_loss_cls_pre, RCNN_loss_bbox_pre, \
                rois_label_pre, feature_pre, rois_feature_pre, rpn_cls_score_pre, \
                rpn_bbox_pred_pre = fasterRCNN_pre(im_data, im_info, gt_boxes_pre, num_boxes_pre, args.open)

                #######current model
                fasterRCNN_cur.train()
                rois_cur, cls_score_cur, bbox_pred_cur, \
                rpn_loss_cls_cur, rpn_loss_box_cur_cur, \
                RCNN_loss_cls_cur, RCNN_loss_bbox_cur, \
                rois_label_cur, feature_cur, rois_feature_cur, rpn_cls_score_cur, \
                rpn_bbox_pred_cur = fasterRCNN_cur(im_data, im_info, gt_boxes_cur, num_boxes_cur, args.open)

                # for rois fm
                num_rois_pre = (rois_label_pre > 0).int().sum()
                num_rois_cur = (rois_label_cur > 0).int().sum()
                if num_rois_cur + num_rois_pre <= 256:
                    rois_joint = torch.zeros([1, 256, 5], dtype=torch.float).cuda()
                    rois_label_joint = torch.zeros([256], dtype=torch.float).cuda()
                    rois_joint[:, :num_rois_pre, :] = rois_pre[:, :num_rois_pre, :]
                    rois_joint[:, num_rois_pre:(num_rois_pre + num_rois_cur), :] = rois_cur[:, :num_rois_cur, :]
                    rois_label_joint[:num_rois_pre] = rois_label_pre[:num_rois_pre]
                    rois_label_joint[num_rois_pre:(num_rois_pre + num_rois_cur)] = rois_label_cur[:num_rois_cur]
                    num_rois_joint = torch.IntTensor([num_rois_cur + num_rois_pre])
                else:
                    rois_joint = torch.cat((rois_pre, rois_cur), 1)
                    rois_label_joint = torch.cat((rois_label_pre, rois_label_pre), 0)
                    num_rois_joint = torch.IntTensor([[num_rois_cur + num_rois_pre]])

                #######model
                fasterRCNN.zero_grad()
                rois, cls_score, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, feature, rois_feature, rpn_cls_score, \
                rpn_bbox_pred, cls_score_positive, bbox_pred_positive = fasterRCNN(im_data, im_info, gt_boxes_joint,
                                                                                   num_boxes_joint, rois_label_joint,
                                                                                   rois_joint, args.open)

                # fm
                mask_batch_pre, norms_pre, mask_batch_cur, norms_cur = mask_gen.__call__(im_data, feature_pre,
                                                                                         gt_boxes_pre, num_boxes_pre,
                                                                                         gt_boxes_cur, num_boxes_cur)
                fm_loss_pre = (torch.pow(feature - feature_pre, 2) * mask_batch_pre).sum() / norms_pre
                fm_loss_cur = (torch.pow(feature - feature_cur, 2) * mask_batch_cur).sum() / norms_cur

                # rois fm
                if num_boxes_pre > 0 and num_boxes_cur > 0:
                    roi_feature_att = proposal_attention(rois_feature)
                    roi_feature_att_joint = torch.cat((rois_feature_pre, rois_feature_cur), 0)
                    roi_feature_att_joint = proposal_attention(roi_feature_att_joint)
                    rois_fm_loss_joint = F.mse_loss(roi_feature_att, roi_feature_att_joint)
                elif num_boxes_pre > 0 and num_boxes_cur == 0:
                    roi_feature_att = proposal_attention(rois_feature)
                    roi_feature_att_pre = proposal_attention(rois_feature_pre)
                    rois_fm_loss_joint = F.mse_loss(roi_feature_att, roi_feature_att_pre)
                else:
                    roi_feature_att = proposal_attention(rois_feature)
                    roi_feature_att_cur = proposal_attention(rois_feature_cur)
                    rois_fm_loss_joint = F.mse_loss(roi_feature_att, roi_feature_att_cur)

                # RCNN cls
                logits_pre = F.softmax(cls_score_pre[:num_rois_pre, :], dim=1)
                logits_cur = F.softmax(cls_score_cur[:num_rois_cur, :], dim=1)

                cls_score_positive = F.softmax(cls_score_positive, dim=1)

                logits_for_pre = torch.zeros([cls_score_positive[:num_rois_pre, :].size(0), args.num_classes_old],
                                             dtype=torch.float).cuda()
                logits_for_pre[:, :1] = cls_score_positive[:num_rois_pre, :][:, :1] + torch.sum(
                    cls_score_positive[:num_rois_pre, :][:, args.num_classes_old:], dim=1).unsqueeze(1)
                logits_for_pre[:, 1:args.num_classes_old] = cls_score_positive[:num_rois_pre, :][:,
                                                            1:args.num_classes_old]

                logits_for_cur = torch.zeros(
                    [cls_score_positive[num_rois_pre:(num_rois_pre + num_rois_cur), :].size(0), args.num_classes_new],
                    dtype=torch.float).cuda()
                logits_for_cur[:, :1] = cls_score_positive[num_rois_pre:(num_rois_pre + num_rois_cur), :][:,
                                        :1] + torch.sum(
                    cls_score_positive[num_rois_pre:(num_rois_pre + num_rois_cur), :][:, 1:args.num_classes_old],
                    dim=1).unsqueeze(1)
                logits_for_cur[:, 1:] = cls_score_positive[num_rois_pre:(num_rois_pre + num_rois_cur), :][:,
                                        args.num_classes_old:]

                RCNN_loss_cls_pre_dis = F.kl_div(torch.log(logits_for_pre), logits_pre, reduction='batchmean')
                RCNN_loss_cls_cur_dis = F.kl_div(torch.log(logits_for_cur), logits_cur, reduction='batchmean')

                # RCNN reg
                reg_pre = bbox_pred_pre[:num_rois_pre, :]
                reg_cur = bbox_pred_cur[:num_rois_cur, :]
                reg_for_pre = bbox_pred_positive[:num_rois_pre, :]
                reg_for_cur = bbox_pred_positive[num_rois_pre:(num_rois_pre + num_rois_cur), :]

                RCNN_loss_bbox_pre_dis = F.smooth_l1_loss(reg_for_pre, reg_pre)
                RCNN_loss_bbox_cur_dis = F.smooth_l1_loss(reg_for_cur, reg_cur)

                ###################################################
                if num_boxes_pre.data > 0 and num_boxes_cur > 0:
                    loss_det = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                    loss_fm = 0.01 * fm_loss_pre.mean() + 0.01 * fm_loss_cur.mean()
                    loss_dis_det = RCNN_loss_cls_pre_dis.mean() + RCNN_loss_cls_cur_dis.mean() + RCNN_loss_bbox_pre_dis.mean() + RCNN_loss_bbox_cur_dis.mean()
                    loss_roi_fm = 300 * rois_fm_loss_joint.mean()
                    loss_temp += loss_det.item()
                    loss_dis_fm_temp += loss_fm.item()
                    loss_dis_det_temp += loss_dis_det.item()
                    loss_roi_fm_temp += loss_roi_fm.item()
                elif num_boxes_pre.data > 0 and num_boxes_cur == 0:
                    loss_det = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                    loss_fm = 0.01 * fm_loss_pre.mean()
                    loss_dis_det = RCNN_loss_cls_pre_dis.mean() + RCNN_loss_bbox_pre_dis.mean()
                    loss_roi_fm = 300 * rois_fm_loss_joint.mean()
                    loss_temp += loss_det.item()
                    loss_dis_fm_temp += loss_fm.item()
                    loss_dis_det_temp += loss_dis_det.item()
                    loss_roi_fm_temp += loss_roi_fm.item()
                else:
                    loss_det = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                    loss_fm = 0.01 * fm_loss_cur.mean()
                    loss_dis_det = RCNN_loss_cls_cur_dis.mean() + RCNN_loss_bbox_cur_dis.mean()
                    loss_roi_fm = 300 * rois_fm_loss_joint.mean()
                    loss_temp += loss_det.item()
                    loss_dis_fm_temp += loss_fm.item()
                    loss_dis_det_temp += loss_dis_det.item()
                    loss_roi_fm_temp += loss_roi_fm.item()

                loss = loss_det + loss_fm + loss_dis_det + loss_roi_fm

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "vgg16":
                    clip_gradient(fasterRCNN, 10.)
                optimizer.step()

                if step % args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (args.disp_interval + 1)
                        loss_dis_fm_temp /= (args.disp_interval + 1)
                        loss_dis_det_temp /= (args.disp_interval + 1)
                        loss_roi_fm_temp /= (args.disp_interval + 1)

                    if args.mGPUs:
                        loss_rpn_cls = rpn_loss_cls.mean().item()
                        loss_rpn_box = rpn_loss_box.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    print(
                        "[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, loss_fm: %.4f, loss_dis_det: %.4f, loss_rois_fm: %.4f, lr: %.2e" \
                        % (args.session, epoch, step, iters_per_epoch, loss_temp, loss_dis_fm_temp, loss_dis_det_temp,
                           loss_roi_fm_temp, lr))  # ,loss_roi_fm, loss_rois_fm: %.4f
                    print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                    if args.use_tfboard:
                        info = {
                            'loss': loss_temp,
                            'loss_dis_fm': loss_dis_fm_temp,
                            'loss_dis_det_fm': loss_dis_det_temp,
                            'loss_roi_fm_temp': loss_roi_fm_temp,
                            'loss_rpn_cls': loss_rpn_cls,
                            'loss_rpn_box': loss_rpn_box,
                            'loss_rcnn_cls': loss_rcnn_cls,
                            'loss_rcnn_box': loss_rcnn_box
                        }
                        logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                           (epoch - 1) * iters_per_epoch + step)

                    loss_temp = 0
                    loss_dis_fm_temp = 0
                    loss_dis_det_temp = 0
                    loss_roi_fm_temp = 0
                    start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
