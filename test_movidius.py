import cv2
import numpy as np
import json
import mvnc.mvncapi as mvncapi
import time
import argparse

import movidus_utils
import yolo_utils

def inference_image(graph_file, meta_file, img_in_name, img_out_name, threshold):
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold
    dev = movidus_utils.get_mvnc_device()
    graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)

    img = cv2.imread(img_in_name)
    img = img.astype(np.float32)
    img_orig = np.copy(img)
    img_orig_dimensions = img_orig.shape
    img = yolo_utils.pre_proc_img(img, meta)
    
    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, 'user object')
    output, user_obj = output_fifo.read_elem()
    print(output.shape)
    print(user_obj)
    
    y_out = np.reshape(output, (13, 13,125))
    y_out = np.squeeze(y_out)
    boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
    print(boxes)
    
    yolo_utils.add_bb_to_img(img_orig, boxes)
    cv2.imwrite(img_out_name, img_orig)

def inference_video(graph_file, meta_file, video_in_name, video_out_name, threshold):
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold   
    dev = movidus_utils.get_mvnc_device()
    graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)
    cap = cv2.VideoCapture()
    cap.open(video_in_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ) 
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out_name, fourcc, fps, (width,height))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Video Ended")
            break   
        frame_orig = np.copy(frame)
        img_orig_dimensions = frame_orig.shape
        frame = yolo_utils.pre_proc_img(frame, meta)
        start = time.time()
        graph.queue_inference_with_fifo_elem(
            input_fifo, output_fifo, frame, 'user object')
        output, _ = output_fifo.read_elem()
        end = time.time()
        print('FPS: ',format(  (1/ (end - start))  ) )        

        y_out = np.reshape(output, (13, 13,125))
        y_out = np.squeeze(y_out)
        # # Posproc
        boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
        yolo_utils.add_bb_to_img(frame_orig, boxes)
        out.write(frame_orig)
    cap.release()
    out.release()
    
def inference_video_test_times(graph_file, meta_file, video_in_name, video_out_name, threshold):
    
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold   
    dev = movidus_utils.get_mvnc_device()
    graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)
    cap = cv2.VideoCapture()
    cap.open(video_in_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ) 
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out_name, fourcc, fps, (width,height))
    times = []
    for i in range(200):
        ret, frame = cap.read()
        if not ret:
            print("Video Ended")
            break   
        frame_orig = np.copy(frame)
        img_orig_dimensions = frame_orig.shape
        frame = yolo_utils.pre_proc_img(frame, meta)
        start = time.time()
        graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, frame, 'user object')
        output, _ = output_fifo.read_elem()
        end = time.time()
        print('Frame: {} FPS: {}'.format(i,  (1/ (end - start))  ) )
        times.append((1/ (end - start)))
        y_out = np.reshape(output, (13, 13,125))
        y_out = np.squeeze(y_out)
        # # Posproc
        boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
        yolo_utils.add_bb_to_img(frame_orig, boxes)
        out.write(frame_orig)
    cap.release()
    out.release()
    print('mean_fps: {}'.format(np.mean(times)))

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
        "-mg", "--movidius_graph", 
        required=False,
        default= 'built_graph/tiny-yolo-voc.graph',
        help="path to movidius graph")
    ap.add_argument(
        "-th", "--threshold", 
        required=False,
        default = 0.3, 
        help="threshold")
    args = ap.parse_args()
    inference_video_test_times(
        args.movidius_graph, 
        args.meta_file, 
        args.input_video, 
        args.output_video, 
        args.threshold)

if __name__ == '__main__':
    main()
