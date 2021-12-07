from __future__ import print_function
import os, sys, zipfile
import json

class_num = 0


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


json_file = "/home/zhangc/PycharmProjects/faster-rcnn-incremental-non-overlaps.pytorch/data/coco/annotations/instances_train2014.json"

data = json.load(open(json_file, 'r'))
ana_txt_save_path = "/home/zhangc/PycharmProjects/faster-rcnn-incremental-non-overlaps.pytorch/data/coco/train2014"
if not os.path.exists(ana_txt_save_path):
    os.makedirs(ana_txt_save_path)

index = 0
cat_id_map = {}
for img in data['images']:
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    ana_txt_name = filename.split(".")[0] + ".txt"
    index += 1
    print(str(index) + '   ' + str(ana_txt_name))
    f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')

    for ann in data['annotations']:
        if ann['image_id'] == img_id:
            if ann['category_id'] not in cat_id_map:
                cat_id_map[ann['category_id']] = class_num
                class_num += 1
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (cat_id_map[ann['category_id']], box[0], box[1], box[2], box[3]))

    f_txt.close()
fileObject = open("/home/zhangc/PycharmProjects/faster-rcnn-incremental-non-overlaps.pytorch/data/coco/train_cat_id_map.txt", 'w')  # 类别进行重新映射
for cat_id in cat_id_map:
    fileObject.write(str(cat_id))
    fileObject.write('\n')
fileObject.close()
