import cv2
import numpy as np
import json
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

def get_meta(meta_file):
    with open(meta_file, 'r') as fp:
        meta = json.load(fp)
    return meta

def pre_proc_img(im, meta):
    w, h, _ = meta['inp_size']
    im = im.astype(np.float32)
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    imsz = np.expand_dims(imsz, axis=0)
    return imsz

def findboxes_meta(meta, net_out):
    boxes = list()
    boxes = box_constructor(meta, net_out)
    return boxes

def process_box_meta(meta, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = meta['labels'][max_indx]
    if max_prob > threshold:
        left = int((b.x - b.w/2.) * w)
        right = int((b.x + b.w/2.) * w)
        top = int((b.y - b.h/2.) * h)
        bot = int((b.y + b.h/2.) * h)
        if left < 0:
            left = 0
        if right > w - 1:
            right = w - 1
        if top < 0:
            top = 0
        if bot > h - 1:
            bot = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bot, mess, max_indx, max_prob)
    return None

def procces_out(out, meta, img_orig_dimensions):
    h, w, _ = img_orig_dimensions
    boxes = findboxes_meta(meta, out)
    threshold = meta['thresh']
    boxesInfo = list()
    for box in boxes:
        tmpBox = process_box_meta(meta, box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

def add_bb_to_img(img_orig, boxes):
    for box in boxes:
        left = box["topleft"]['x']
        right = box["bottomright"]['x']
        top = box["topleft"]['y']
        bot = box["bottomright"]['y']
        mess = box["label"]
        confidence = box["confidence"]
        h, w, _ = img_orig.shape
        thick = int((h + w) // 300)
        cv2.rectangle(img_orig,
                      (left, top), (right, bot),
                      [255, 0, 0], thick)
        cv2.putText(
            img_orig, mess, (left, top - 12),
            0, 1e-3 * h, [255, 0, 0],
            thick // 3)