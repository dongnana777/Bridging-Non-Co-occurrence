"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb
import torchvision as tv
import torch

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in range(imdb.num_images)]

  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
  # rank roidb based on the ratio between width and height.
  ratio_large = 2  # largest ratio to preserve.
  ratio_small = 0.5  # smallest ratio to preserve.

  ratio_list = []
  for i in range(len(roidb)):
    width = roidb[i]['width']
    height = roidb[i]['height']
    ratio = width / float(height)

    if ratio > ratio_large:
      roidb[i]['need_crop'] = 1
      ratio = ratio_large
    elif ratio < ratio_small:
      roidb[i]['need_crop'] = 1
      ratio = ratio_small
    else:
      roidb[i]['need_crop'] = 0

    ratio_list.append(ratio)

  ratio_list = np.array(ratio_list)
  ratio_index = np.argsort(ratio_list)
  return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
  # filter the image without bounding box.
  print('before filtering, there are %d images...' % (len(roidb)))
  i = 0
  while i < len(roidb):
    if len(roidb[i]['boxes']) == 0:
      del roidb[i]
      i -= 1
    i += 1
  print('after filtering, there are %d images...' % (len(roidb)))
  return roidb

def random_choice(roidb):
  print('before choice, there are %d images...' % (len(roidb)))
  roidb = np.random.choice(roidb, size=15000)
  print('after choice, there are %d images...' % (len(roidb)))
  return roidb

def filter_img_id(roidb, img_id, gt_boxes_pre, gt_boxes_cur,gt_boxes_num):
  # filter the image not in img_id.
  print('before filtering, there are %d images...' % (len(roidb)))
  i = 0
  while i < len(roidb):
    if roidb[i]['img_id'] not in img_id:
      del roidb[i]
      i -= 1
    i += 1
  print('after filtering, there are %d images...' % (len(roidb)))

  i = 0
  while i < len(roidb):
    for j in range(len(img_id)):
      if roidb[i]['img_id'] == img_id[j]:
        roidb[i]['pseudo_gt_boxes_pre'] = gt_boxes_pre[j]
        roidb[i]['pseudo_gt_boxes_cur'] = gt_boxes_cur[j]
        roidb[i]['pseudo_gt_boxes_num'] = gt_boxes_num[j]
    i += 1

  return roidb

def combined_roidb(imdb_names, labels=None, labels_old=None, img_selected=None, gt_boxes_pre=None,gt_boxes_cur=None, gt_boxes_num=None, choice=True, training=True):
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)
    print('done')

    return imdb.roidb

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}`'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)

  if training:
    roidb = filter_roidb(roidb)
    if labels_old is not None:
        roidb = filter_images(roidb, labels, labels_old)
    if img_selected is not None:
      roidb = filter_img_id(roidb, img_selected, gt_boxes_pre, gt_boxes_cur, gt_boxes_num)
      roidb = random_choice(roidb)

  ratio_list, ratio_index = rank_roidb_ratio(roidb)

  return imdb, roidb, ratio_list, ratio_index

def __strip_zero(labels):
  while 0 in labels:
    labels.remove(0)

def filter_images(roidb,
                  labels_new=None,
                  labels_old=None,
                  train=True):

  if labels_new is not None:
    # store the labels
    labels_old = labels_old if labels_old is not None else []

    __strip_zero(labels_new)
    __strip_zero(labels_old)

    order = [0] + labels_old + labels_new
    ##################################
    if train:
      masking_value = 0
    else:
      masking_value = 255

    inverted_order = {label: order.index(label) for label in order}
    inverted_order[255] = masking_value

    target_transform = tv.transforms.Lambda(lambda t: t.apply_(lambda x: inverted_order[x] if x in inverted_order else masking_value))

    #####################################
    if 0 in labels_new:
      labels_new.remove(0)
    print('Filtering images')

    fil = lambda c: any(x in labels_new for x in cls)  #for old or new model
    # fil = lambda c: all(x not in labels_new for x in cls)  # for old or new model on no category images
    # fil = lambda c: all(x in labels_new for x in cls) or all(x in labels_old for x in cls) # for non-overlaps joint-training

    i = 0
    while i < len(roidb):
      cls = roidb[i]['gt_classes']
      if fil(cls):
        roidb[i]['gt_classes'] = target_transform(torch.tensor(cls)).detach().numpy() #cls
      else:
        del roidb[i]
        i -= 1
      i += 1
    print('Done')
    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb
