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



class mask_gen(nn.Module):
    def __init__(self, args):
        super(mask_gen, self).__init__()
        self.args = args
    def __call__(self, im_data, feature, gt_boxes_pre, num_boxes_pre, gt_boxes_cur,num_boxes_cur):
        height, width = feature.size(2), feature.size(3)
        height_img, width_img = im_data.size(2), im_data.size(3)

        if num_boxes_pre.data > 0:
            ####### gt_pre_mask
            mask_batch_pre = []
            mask_x = torch.zeros([height, width], dtype=torch.int64).cuda()
            mask_y = torch.zeros([height, width], dtype=torch.int64).cuda()
            mask_image = torch.zeros([height, width], dtype=torch.int64).cuda()
            for i in gt_boxes_pre.squeeze(0):
                if i[-1] != 0:
                    i = i.unsqueeze(0)
                    spatial_scale = float(height) / height_img
                    x1 = torch.trunc(i[:, 0] * spatial_scale)
                    y1 = torch.trunc(i[:, 1] * spatial_scale)
                    x2 = torch.trunc(i[:, 2] * spatial_scale)
                    y2 = torch.trunc(i[:, 3] * spatial_scale)

                    mask_y[y1.int():y2.int(), :] = 1
                    num1 = mask_y.sum()

                    mask_x[:, x1.int():x2.int()] = 1
                    num2 = mask_x.sum()

                    mask = ((mask_x + mask_y) > 1)
                    num3 = mask.sum()

                    mask_image += mask.long()
            mask_batch_pre.append(mask_image)
            ########fm
            mask_list = []
            for mask in mask_batch_pre:
                mask = (mask > 0).float().unsqueeze(0)
                mask_list.append(mask)
            mask_batch_pre = torch.stack(mask_list, dim=0)
            norms_pre = mask_batch_pre.sum() * 2
            if norms_pre == 0:
                norms_pre = 1
        else:
            mask_batch_pre = torch.zeros([1, 1, height, width], dtype=torch.int64).cuda()
            norms_pre = 1

        if num_boxes_cur.data > 0:
            ####### gt_pre_mask
            mask_batch_cur = []
            mask_x = torch.zeros([height, width], dtype=torch.int64).cuda()
            mask_y = torch.zeros([height, width], dtype=torch.int64).cuda()
            mask_image = torch.zeros([height, width], dtype=torch.int64).cuda()
            for i in gt_boxes_cur.squeeze(0):
                if i[-1] != 0:
                    i = i.unsqueeze(0)
                    spatial_scale = float(height) / height_img
                    x1 = torch.trunc(i[:, 0] * spatial_scale)
                    y1 = torch.trunc(i[:, 1] * spatial_scale)
                    x2 = torch.trunc(i[:, 2] * spatial_scale)
                    y2 = torch.trunc(i[:, 3] * spatial_scale)

                    mask_y[y1.int():y2.int(), :] = 1
                    num1 = mask_y.sum()

                    mask_x[:, x1.int():x2.int()] = 1
                    num2 = mask_x.sum()

                    mask = ((mask_x + mask_y) > 1)
                    num3 = mask.sum()

                    mask_image += mask.long()
            mask_batch_cur.append(mask_image)
            ########fm
            mask_list = []
            for mask in mask_batch_cur:
                mask = (mask > 0).float().unsqueeze(0)
                mask_list.append(mask)
            mask_batch_cur = torch.stack(mask_list, dim=0)
            norms_cur = mask_batch_cur.sum() * 2
            if norms_cur == 0:
                norms_cur = 1
        else:
            mask_batch_cur = torch.zeros([1, 1, height, width], dtype=torch.int64).cuda()
            norms_cur = 1

        return mask_batch_pre, norms_pre, mask_batch_cur, norms_cur


def normalize_atten_maps(atten_maps):
    atten_shape = atten_maps.size()
    batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                             batch_maxs - batch_mins + 1e-7)
    atten_normed = atten_normed.view(atten_shape)
    return atten_normed

def proposal_attention(pooled_feat):
    rois_self_attention = torch.mean(pooled_feat, dim=1)
    rois_self_attention = torch.sigmoid(normalize_atten_maps(rois_self_attention))
    return rois_self_attention

