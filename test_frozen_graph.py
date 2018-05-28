import cv2
import numpy as np
import tensorflow as tf
import json
import time

from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

def get_meta(meta_file):
    with open(meta_file, 'r') as fp:
        meta = json.load(fp)
    return meta

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def pre_proc_img(im, meta):
    w, h, _ = meta['inp_size']
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    imsz = np.expand_dims(imsz, axis=0)
    return imsz

def findboxes_meta(meta, net_out):
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

def process_box_meta(meta, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = meta['labels'][max_indx]
    if max_prob > threshold:
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bot, mess, max_indx, max_prob)
    return None

def procces_out(out, meta, img_orig_dimensions):
    h, w, _ = img_orig_dimensions
    boxes = findboxes_meta(meta,out) 
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

def inference_single_image(pb_file, meta_file, img_test_path, img_out_name, threshold):
    meta = get_meta(meta_file)
    meta['thresh'] = threshold
    graph = load_graph(pb_file)
    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/output:0')

    with tf.Session(graph=graph) as sess:
        # Preproc    
        img = cv2.imread(img_test_path)
        img_orig = np.copy(img)
        img_orig_dimensions = img_orig.shape
        img = pre_proc_img(img, meta)
        # inference
        y_out = sess.run(y,feed_dict={x:img})
        # Posproc
        y_out = np.squeeze(y_out)
        boxes = procces_out(y_out, meta, img_orig_dimensions)
        print(boxes)
        add_bb_to_img(img_orig, boxes)
        cv2.imwrite(img_out_name, img_orig)

def inference_video(pb_file, meta_file, video_in, video_out, threshold):
    meta = get_meta(meta_file)
    meta['thresh'] = threshold

    graph = load_graph(pb_file)
    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/output:0')

    cap = cv2.VideoCapture()
    cap.open(video_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ) 
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out, fourcc, fps, (width,height))

    with tf.Session(graph=graph) as sess:
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Video Ended")
                break   
            frame_orig = np.copy(frame)
            img_orig_dimensions = frame_orig.shape
            frame = pre_proc_img(frame, meta)
            # inference
            

            start = time.time()
            y_out = sess.run(y,feed_dict={x:frame})            
            end = time.time()
            print('FPS: ',format(  (1/ (end - start))  ) )
            
            # Posproc
            y_out = np.squeeze(y_out)
            boxes = procces_out(y_out, meta, img_orig_dimensions)
            # print boxes
            add_bb_to_img(frame_orig, boxes)
            print(frame_orig.shape)
            out.write(frame_orig)
    cap.release()
    out.release()
    

def inference_video_test_times(pb_file, meta_file, video_in, video_out, threshold):
    meta = get_meta(meta_file)
    meta['thresh'] = threshold

    graph = load_graph(pb_file)
    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/output:0')

    cap = cv2.VideoCapture()
    cap.open(video_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ) 
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out, fourcc, fps, (width,height))

    times = []
    with tf.Session(graph=graph) as sess:
        for i in range(200):
            ret, frame = cap.read()
            if not ret:
                print("Video Ended")
                break   
            frame_orig = np.copy(frame)
            img_orig_dimensions = frame_orig.shape
            frame = pre_proc_img(frame, meta)
            # inference
            start = time.time()
            y_out = sess.run(y,feed_dict={x:frame})            
            end = time.time()
            print('Frame: {} FPS: {}'.format(i,  (1/ (end - start))  ) )
            times.append((1/ (end - start)))


            # Posproc
            y_out = np.squeeze(y_out)
            boxes = procces_out(y_out, meta, img_orig_dimensions)
            # print boxes
            add_bb_to_img(frame_orig, boxes)
            print(frame_orig.shape)
            out.write(frame_orig)
    cap.release()
    out.release()

    print('mean_fps: {}'.format(np.mean(times)))



pb_file = 'built_graph/yolov2-tiny-voc.pb'
meta_file = 'built_graph/yolov2-tiny-voc.meta'

# img_test_path = 'sample_img/sample_dog.jpg'
# img_out_name = 'test_out.jpg'
# threshold = 0.2
# inference_single_image(pb_file, meta_file, img_test_path, img_out_name, threshold)
video_in = '/datos/object-detection/videos_to_test/manana.avi'
video_out = 'test_out.avi'
threshold = 0.3
#inference_video(pb_file, meta_file, video_in, video_out, threshold)
inference_video_test_times(pb_file, meta_file, video_in, video_out, threshold)

