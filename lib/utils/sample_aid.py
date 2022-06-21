import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import h5py
import PIL.ImageFile
from PIL import Image
import torchvision.transforms as transforms
import time
import types
from numpy import random
from pycocotools.coco import COCO
from datasets.coco import *
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from augmentation_np import *


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def tensor_to_np_bbox(bbox):
    bbox = bbox.cpu().numpy()
    return bbox

def np_to_tensor_bbox(bbox):
    assert type(bbox) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(bbox))
    bbox = torch.from_numpy(bbox).cuda().float()
    return bbox

def box_voting(all_dets, thresh=0.3, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]

    # nms
    cls_scores = all_dets[:, 5]
    cls_dets = all_dets[:, :4]
    _, order = torch.sort(cls_scores, 0, True)
    keep = nms(cls_dets[order, :], cls_scores[order], cfg.TEST.NMS)
    top_dets = all_dets[keep.view(-1).long()]

    top_dets = tensor_to_np_bbox(top_dets)
    all_dets = tensor_to_np_bbox(all_dets)

    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 5]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws ** beta) ** (1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws)) ** beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )
    # nms
    top_dets_out = np_to_tensor_bbox(top_dets_out)

    cls_scores = top_dets_out[:, 5]
    cls_dets = top_dets_out[:, :4]
    _, order = torch.sort(cls_scores, 0, True)
    keep = nms(cls_dets[order, :], cls_scores[order], cfg.TEST.NMS)
    top_dets_out_all = top_dets_out[keep.view(-1).long()]

    # unique
    top_dets_out_all = tensor_to_np_bbox(top_dets_out_all[:, :5])
    top_dets_out_all = np.unique(top_dets_out_all, axis=0)
    top_dets_out_all = np_to_tensor_bbox(top_dets_out_all)
    return top_dets_out_all


class sample_func(nn.Module):
    def __init__(self, args):
        super(sample_func, self).__init__()
        self.args = args

    def __call__(self, cls_prob, rois, bbox_pred, im_info, per_class_threshold, num_classes):
        img_id_selected = []
        pseudo_gt_boxes_pre = []
        num_boxes_pre = torch.LongTensor(self.args.batch_size).zero_().cuda()
        for i in range(self.args.batch_size):
            gt_boxes_pre = []
            all_boxes = []
            scores = cls_prob[i].unsqueeze(0).data
            boxes = rois[i].unsqueeze(0).data[:, :, 1:5]
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred[i].unsqueeze(0).data

            # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * num_classes)  # len(imdb.classes)  args.num_classes_old

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info[i].unsqueeze(0).data, 1)

            # pred_boxes /= im_info[0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            for j in xrange(1, num_classes):  # imdb.num_classes args.num_classes_old
                thresh = per_class_threshold[j]
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)

                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls = torch.full([cls_boxes.size(0), 1], j).cuda()
                    cls_dets_ = torch.cat((cls_boxes, cls), 1)
                    cls_dets_ = cls_dets_[order]

                    cls_dets = torch.cat((cls_boxes, cls), 1)
                    cls_dets = torch.cat((cls_dets, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]

                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)

                    # cls_dets = cls_dets[keep.view(-1).long()]
                    cls_dets_ = cls_dets_[keep.view(-1).long()]

                    gt_boxes_pre.append(cls_dets_)
                    all_boxes.append(cls_dets)

            if len(gt_boxes_pre) != 0:
                gt_boxes_pre = torch.cat([o for o in gt_boxes_pre], 0)
                if gt_boxes_pre.size(0) > 0 and gt_boxes_pre.size(0) < self.args.num_classes:
                    num_boxes_pre[i] = torch.tensor([gt_boxes_pre.size(0)])
                    zero_tensor = torch.zeros([(self.args.num_classes - gt_boxes_pre.size(0)), 5], dtype=torch.float).cuda()
                    gt_boxes_pre_zeropad = torch.cat((gt_boxes_pre, zero_tensor), 0).unsqueeze(0)
                elif gt_boxes_pre.size(0) == self.args.num_classes:
                    num_boxes_pre[i] = torch.tensor([self.args.num_classes])
                    gt_boxes_pre_zeropad = gt_boxes_pre.unsqueeze(0)
                else:
                    num_boxes_pre[i] = torch.tensor([self.args.num_classes])
                    gt_boxes_pre_zeropad = gt_boxes_pre[:self.args.num_classes, :].unsqueeze(0)
            else:
                num_boxes_pre[i] = torch.tensor([0])
                gt_boxes_pre_zeropad = torch.zeros([1, self.args.num_classes, 5], dtype=torch.float).cuda()

            pseudo_gt_boxes_pre.append(gt_boxes_pre_zeropad.cuda())
        pseudo_gt_boxes_pre = torch.cat([o for o in pseudo_gt_boxes_pre], 0)

        return pseudo_gt_boxes_pre, num_boxes_pre, all_boxes

    def _pad_gt_boxes(self, gt_boxes_pre):
        pseudo_gt_boxes_pre = []
        for i in range(self.args.batch_size):
            num_boxes_pre = torch.LongTensor(self.args.batch_size).zero_().cuda()
            if gt_boxes_pre.size(0) > 0 and gt_boxes_pre.size(0) < self.args.num_classes:
                num_boxes_pre[i] = torch.tensor([gt_boxes_pre.size(0)])
                zero_tensor = torch.zeros([(self.args.num_classes - gt_boxes_pre.size(0)), 5], dtype=torch.float).cuda()
                gt_boxes_pre_zeropad = torch.cat((gt_boxes_pre, zero_tensor), 0).unsqueeze(0)
            elif gt_boxes_pre.size(0) == self.args.num_classes:
                num_boxes_pre[i] = torch.tensor([self.args.num_classes])
                gt_boxes_pre_zeropad = gt_boxes_pre.unsqueeze(0)
            else:
                num_boxes_pre[i] = torch.tensor([self.args.num_classes])
                gt_boxes_pre_zeropad = gt_boxes_pre[:self.args.num_classes, :].unsqueeze(0)
            pseudo_gt_boxes_pre.append(gt_boxes_pre_zeropad.cuda())
        pseudo_gt_boxes_pre = torch.cat([o for o in pseudo_gt_boxes_pre], 0)
        return pseudo_gt_boxes_pre


class sample_pseudo_gt_boxes(nn.Module):

    def __init__(self, args):
        super(sample_pseudo_gt_boxes, self).__init__()
        self.args = args

    def __call__(self, net, im_data, im_info, gt_boxes, num_boxes, num_classes, per_class_threshold):

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, feature = net(im_data, im_info, gt_boxes, num_boxes, self.args.open)
        pseudo_gt_boxes, num_boxes, all_boxes = sample_func(self.args).__call__(cls_prob, rois, bbox_pred,
                                                                     im_info, per_class_threshold, num_classes)

        img_flip = flip_img(im_data, 1)
        rois_flip, cls_prob_flip, bbox_pred_flip, \
        rpn_loss_cls_flip, rpn_loss_box_flip, \
        RCNN_loss_cls_flip, RCNN_loss_bbox_flip, \
        rois_label_flip, feature_flip = net(img_flip, im_info, gt_boxes, num_boxes, self.args.open)
        pseudo_gt_boxes_flip, num_boxes_flip, all_boxes_flip = sample_func(self.args).__call__(cls_prob_flip, rois_flip,
                                                                                    bbox_pred_flip,
                                                                                    im_info,
                                                                                    per_class_threshold,
                                                                                    num_classes)

        img_scaling = resize_img(im_data, (0, 0), fx=0.7, fy=0.7)
        rois_scaling, cls_prob_scaling, bbox_pred_scaling, \
        rpn_loss_cls_scaling, rpn_loss_box_scaling, \
        RCNN_loss_cls_scaling, RCNN_loss_bbox_scaling, \
        rois_label_scaling, feature_scaling = net(img_scaling, im_info, gt_boxes, num_boxes, self.args.open)
        pseudo_gt_boxes_scaling, num_boxes_scaling, all_boxes_scaling = sample_func(self.args).__call__(cls_prob_scaling,
                                                                                             rois_scaling,
                                                                                             bbox_pred_scaling,
                                                                                             im_info,
                                                                                             per_class_threshold,
                                                                                             num_classes)


        if num_boxes >0 or num_boxes_flip > 0 or num_boxes_scaling > 0:
            # all category>5
            category_ori = []
            for j in range(len(all_boxes)):
                    category_ori.append(all_boxes[j][:1, 4])
            category_flip = []
            for j in range(len(all_boxes_flip)):
                    category_flip.append(all_boxes_flip[j][:1, 4])
            category_scaling = []
            for j in range(len(all_boxes_scaling)):
                    category_scaling.append(all_boxes_scaling[j][:1, 4])
            # select category
            category_selected_ = category_ori
            for i in category_flip:
                if i not in category_ori:
                    category_selected_.append(i)
            category_selected = category_selected_
            for i in category_scaling:
                if i not in category_selected_:
                    category_selected.append(i)
            # print(category_selected)
            # select boxes
            if len(category_selected) > 0:
                all_boxes_ = []
                all_boxes_flip_ = []
                all_boxes_scaling_ = []

                for i in range(len(all_boxes)):
                    if all_boxes[i][:1, 4] in category_selected:
                            all_boxes_.append(all_boxes[i])
                for i in range(len(all_boxes_flip)):
                    if all_boxes_flip[i][:1, 4] in category_selected:
                            all_boxes_flip_.append(all_boxes_flip[i])
                for i in range(len(all_boxes_scaling)):
                    if all_boxes_scaling[i][:1, 4] in category_selected:
                            all_boxes_scaling_.append(all_boxes_scaling[i])

                # boxes voting
                voted_boxes = []
                for cat in category_selected:
                    all_boxes_per_class = []
                    all_boxes_flip_per_class = []
                    all_boxes_scaling_per_class = []
                    for j in range(len(all_boxes_)):
                        if all_boxes_[j][:1, 4] == cat:
                            all_boxes_per_class.append(all_boxes_[j])
                    for j in range(len(all_boxes_flip_)):
                        if all_boxes_flip_[j][:1, 4] == cat:
                            all_boxes_flip_per_class.append(flip_box(all_boxes_flip_[j], img_flip.size(3), img_flip.size(2), 1))
                    for j in range(len(all_boxes_scaling_)):
                        if all_boxes_scaling_[j][:1, 4] == cat:
                            all_boxes_scaling_per_class.append(resize_box(all_boxes_scaling_[j], fx=float(1 / 0.7), fy=float(1 / 0.7)))

                    if len(all_boxes_per_class) >0 and len(all_boxes_flip_per_class) >0 and len(all_boxes_scaling_per_class) >0 :
                        voted_boxes_per_class = torch.cat((all_boxes_per_class, all_boxes_flip_per_class, all_boxes_scaling_per_class), 0)
                    elif len(all_boxes_per_class) == 0 and len(all_boxes_flip_per_class) >0 and len(all_boxes_scaling_per_class) >0 :
                        voted_boxes_per_class = torch.cat((all_boxes_flip_per_class, all_boxes_scaling_per_class), 0)
                    elif len(all_boxes_per_class) >0 and len(all_boxes_flip_per_class) ==0 and len(all_boxes_scaling_per_class) >0 :
                        voted_boxes_per_class = torch.cat((all_boxes_per_class, all_boxes_scaling_per_class), 0)
                    elif len(all_boxes_per_class) >0 and len(all_boxes_flip_per_class) >0 and len(all_boxes_scaling_per_class) ==0 :
                        voted_boxes_per_class = torch.cat((all_boxes_per_class, all_boxes_flip_per_class), 0)
                    elif len(all_boxes_per_class) ==0 and len(all_boxes_flip_per_class) ==0 and len(all_boxes_scaling_per_class) >0 :
                        voted_boxes_per_class = all_boxes_scaling_per_class
                    elif len(all_boxes_per_class) ==0 and len(all_boxes_flip_per_class) >0 and len(all_boxes_scaling_per_class) ==0 :
                        voted_boxes_per_class = all_boxes_flip_per_class
                    elif len(all_boxes_per_class) >0 and len(all_boxes_flip_per_class) ==0 and len(all_boxes_scaling_per_class) ==0 :
                        voted_boxes_per_class = all_boxes_per_class
                    voted_boxes.append(box_voting(voted_boxes_per_class))
                voted_pseudo_gt_boxes = torch.cat([o for o in voted_boxes], 0)
                num_pseudo_gt_boxes = torch.tensor([voted_pseudo_gt_boxes.size(0)])
                voted_pseudo_gt_boxes = sample_func(self.args)._pad_gt_boxes(voted_pseudo_gt_boxes)
                # print(voted_pseudo_gt_boxes)
            else:
                voted_pseudo_gt_boxes = torch.zeros([1,self.args.num_classes, 5], dtype=torch.float).cuda()
                num_pseudo_gt_boxes = torch.tensor([0])
        else:
            voted_pseudo_gt_boxes = torch.zeros([1, self.args.num_classes, 5], dtype=torch.float).cuda()
            num_pseudo_gt_boxes = torch.tensor([0])

        return voted_pseudo_gt_boxes, num_pseudo_gt_boxes
