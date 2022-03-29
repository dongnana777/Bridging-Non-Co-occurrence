import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, rois_label_positive, rois_positive, open):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map to RPN to obtain rois
        # [1,2000,5]
        rois, rpn_loss_cls, rpn_loss_bbox, rpn_cls_score, rpn_bbox_pred, mask = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            # [1,256,5],[1,256],[1,256,4],[1,256,4],[1,256,4]
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, rois_positive_selected = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_label_positive = Variable(rois_label_positive.view(-1).long())
            # num = (rois_label > 0).int().sum()
            # rois_label_positive = Variable(rois_label.view(-1).long()[:num])
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        if self.training:
            # rois_positive = Variable(rois[:, :num, :])
            rois_positive = Variable(rois_positive)
        if self.training and num_boxes > 0:
            rois_positive_selected = Variable(rois_positive_selected)
        else:
            pooled_feat_positive_selected= None

        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            if self.training :
                pooled_feat_positive = self.RCNN_roi_align(base_feat, rois_positive.view(-1, 5))
            if self.training and num_boxes > 0:
                pooled_feat_positive_selected = self.RCNN_roi_align(base_feat, rois_positive_selected.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            if self.training :
                pooled_feat_positive = self.RCNN_roi_align(base_feat, rois_positive.view(-1, 5))
            if self.training and num_boxes > 0:
                pooled_feat_positive_selected = self.RCNN_roi_align(base_feat, rois_positive_selected.view(-1, 5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  # [256,4096]
        if self.training:
            pooled_feat_positive = self._head_to_tail(pooled_feat_positive)  # [256,4096]

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)  # [256,84]
        if self.training:
            bbox_pred_positive = self.RCNN_bbox_pred(pooled_feat_positive)  # [256,84]

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4).long())
            bbox_pred_ = bbox_pred_select.squeeze(1)  # [256,4]
            if self.training:
                bbox_pred_view_positive = bbox_pred_positive.view(bbox_pred_positive.size(0), int(bbox_pred_positive.size(1) / 4), 4)
                bbox_pred_select_positive = torch.gather(bbox_pred_view_positive, 1, rois_label_positive.view(rois_label_positive.size(0), 1, 1).expand(rois_label_positive.size(0), 1, 4))
                bbox_pred_positive = bbox_pred_select_positive.squeeze(1)  # [256,4]

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)  # [256,21]
        if self.training:
            cls_score_positive = self.RCNN_cls_score(pooled_feat_positive)
            cls_prob_positive = F.softmax(cls_score_positive, 1)  # [256,21]

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)  # [256,21],[256]
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred_, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)  # [1,256,21]
        bbox_pred = bbox_pred_.view(batch_size, rois.size(1), -1)  # [1,256,4]

        if self.training:
            return rois, cls_score, bbox_pred_, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox,\
                   rois_label, base_feat, pooled_feat_positive_selected, rpn_cls_score, rpn_bbox_pred, cls_score_positive, bbox_pred_positive
        else:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, base_feat

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
