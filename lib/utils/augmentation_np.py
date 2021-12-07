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
def feature_augumentation(mask_batch):
    mask_list_0 = []
    for mask in mask_batch[0]:
        # max= torch.max(mask)
        mask = (mask > 0).float()
        mask_list_0.append(mask)
    mask_batch_0 = torch.stack(mask_list_0, dim=0).squeeze(0)

    mask_list_1 = []
    for mask in mask_batch[1]:
        # max= torch.max(mask)
        mask = (mask > 0).float()
        mask_list_1.append(mask)
    mask_batch_1 = torch.stack(mask_list_1, dim=0).squeeze(0)
    mask = mask_batch_0 - mask_batch_1

    return mask


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]



def jaccard_numpy(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class RandomHorizontalFlip(object):
    def __init__(self, p=1):
        self.p = p
    def __call__(self, img, bboxes, labels):
        img = np.fliplr(img).copy()
        return img, bboxes, labels

class Resize(object):
    def __init__(self, size_ratio=1):
        self.size_ratio = size_ratio

    def __call__(self, image, boxes=None, labels=None):
        # h, w, _ = image.shape
        image = cv2.resize(image, fx=self.size_ratio, fy=self.size_ratio )
        return image, boxes, labels
class Un_Resize_boxes(object):
    def __init__(self, size_ratio=1):
        self.size_ratio = size_ratio
    def __call__(self, image, boxes=None, labels=None):
        boxes = boxes/self.size_ratio
        return image, boxes, labels

# class SSDAugmentation(object):
#     def __init__(self, size=300, mean=(104, 117, 123)):
#         self.mean = mean
#         self.size = size
#         self.augment = Compose([
#             ConvertFromInts(),
#             ToAbsoluteCoords(),
#             PhotometricDistort(),
#             Expand(self.mean),
#             RandomSampleCrop(),
#             RandomMirror(),
#             ToPercentCoords(),
#             Resize(self.size),
#             SubtractMeans(self.mean)
#         ])
#
#     def __call__(self, img, boxes, labels):
#         return self.augment(img, boxes, labels)

def image_augumentation(image, boxes=None, labels=None):
    augment = Compose([
        # RandomHorizontalFlip(1)
        Resize(size_ratio=0.8)
        # RandomSampleCrop()
        # RandomMirror()
        # PhotometricDistort()
        # RandomLightingNoise(),
        # RandomBrightness(),
        # RandomSaturation(),
        # RandomContrast()
    ])

    image_numpy = tensor_to_np(image)
    boxes_numpy = tensor_to_np_bbox(boxes)
    image_numpy, boxes_numpy, labels = augment(image_numpy, boxes_numpy, labels)
    image_tensor = np_to_tensor(image_numpy)
    boxes_numpy = np_to_tensor_bbox(boxes_numpy)

    return image_tensor, boxes_numpy, labels
def flip_img(src,flip_type):
    src = tensor_to_np(src)
    fliped_img = cv2.flip(src,flip_type)
    fliped_img = np_to_tensor(fliped_img)
    return fliped_img

def flip_xy(x, y, imgw, imgh, flip_type):
    if 1 == flip_type:
        fliped_x = imgw - x
        fliped_y = y
    elif 0 == flip_type:
        fliped_x = x
        fliped_y = imgh - y
    elif -1 == flip_type:
        fliped_x = imgw - x
        fliped_y = imgh - y
    else:
        print('flip type err')
        return
    return fliped_x, fliped_y


def flip_box(boxes,imgw, imgh, flip_type):
    num = boxes.size(0)
    boxes = tensor_to_np_bbox(boxes)
    fliped_boxes = np.zeros([num, 6])
    for i in range(num):
        box = boxes[i]
        x1, y1 = flip_xy(box[0], box[1], imgw, imgh, flip_type)
        x2, y2 = flip_xy(box[2], box[3], imgw, imgh, flip_type)
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        fliped_box = [xmin, ymin, xmax, ymax]
        fliped_boxes[i, :4] = fliped_box
        fliped_boxes[i, 4:] = boxes[i, 4:]

    fliped_boxes = np_to_tensor_bbox(fliped_boxes)
    fliped_boxes = (fliped_boxes).float()
    return fliped_boxes


def resize_xy(x, y, fx, fy):
    return int(x * fx), int(y * fy)


def resize_box(boxes, fx, fy):
    num = boxes.size(0)
    boxes = tensor_to_np_bbox(boxes)
    sized_boxes = np.zeros([num ,6])
    for i in range(num):
        box= boxes[i]
        xmin, ymin = resize_xy(box[0], box[1], fx, fy)
        xmax, ymax = resize_xy(box[2], box[3], fx, fy)
        sized_box = [xmin, ymin, xmax, ymax]
        sized_boxes[i, :4] = sized_box
        sized_boxes[i, 4:] = boxes[i, 4:]
    sized_boxes = np_to_tensor_bbox(sized_boxes)
    sized_boxes = (sized_boxes).float()
    return sized_boxes


def resize_img(src, dsize=(0, 0), fx=1.0, fy=1.0):
    src = tensor_to_np(src)
    sized_img = cv2.resize(src, dsize, fx=fx, fy=fy)
    sized_img = np_to_tensor(sized_img)
    return sized_img



def np_to_tensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0).cuda()
    return img

def tensor_to_np(img):
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

# def np_to_tensor_bbox(bbox):
#     assert type(bbox) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(bbox))
#     bbox = torch.from_numpy(bbox).unsqueeze(0).cuda()
#     return bbox
#
# def tensor_to_np_bbox(bbox):
#     bbox = bbox.cpu().numpy().squeeze(0)
#     return bbox

def np_to_tensor_bbox(bbox):
    assert type(bbox) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(bbox))
    bbox = torch.from_numpy(bbox).cuda()
    return bbox

def tensor_to_np_bbox(bbox):
    bbox = bbox.cpu().numpy()
    return bbox