import json
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf
import yolo_utils

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def inference_single_image(pb_file, 
                           meta_file, 
                           img_test_path, 
                           img_out_name, 
                           threshold):
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold
    graph = load_graph(pb_file)
    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/output:0')
    with tf.Session(graph=graph) as sess:
        img = cv2.imread(img_test_path)
        img_orig = np.copy(img)
        img_orig_dimensions = img_orig.shape
        img = yolo_utils.pre_proc_img(img, meta)
        y_out = sess.run(y,feed_dict={x:img})
        y_out = np.squeeze(y_out)
        boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
        yolo_utils.add_bb_to_img(img_orig, boxes)
        cv2.imwrite(img_out_name, img_orig)
    
def inference_video(pb_file, 
                    meta_file,
                    video_in, 
                    video_out, 
                    threshold):
    meta = yolo_utils.get_meta(meta_file)
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
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video Ended")
                break   
            frame_orig = np.copy(frame)
            img_orig_dimensions = frame_orig.shape
            frame = yolo_utils.pre_proc_img(frame, meta)
            # inference
            start = time.time()
            y_out = sess.run(y,feed_dict={x:frame})            
            end = time.time()
            print('FPS: {:.2f}'.format((1 / (end - start))))
            times.append((1/ (end - start)))
            y_out = np.squeeze(y_out)
            boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
            yolo_utils.add_bb_to_img(frame_orig, boxes)
            out.write(frame_orig)
    cap.release()
    out.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input_video",
        required=True,
        help="path to input video")
    ap.add_argument(
        "-o", "--output_video",
        required=False,
        default='out.avi',
        help="path to output video")
    ap.add_argument(
        "-m", "--meta_file",
        required=False,
        default='built_graph/tiny-yolo-voc.meta',
        help="path to meta file")
    ap.add_argument(
        "-pb", "--pb_file",
        required=False,
        default='built_graph/tiny-yolo-voc.pb',
        help="path to movidius graph")
    ap.add_argument(
        "-th", "--threshold",
        required=False,
        default=0.3,
        help="threshold")
    args = ap.parse_args()
    
    inference_video(
        args.pb_file, 
        args.meta_file, 
        args.input_video, 
        args.output_video,
        args.threshold)

if __name__ == '__main__':
    main()
