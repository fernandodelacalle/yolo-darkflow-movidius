import cv2
import numpy as np
import movidus_utils
import yolo_utils
import time 

class VideoCamera(object):
    def __init__(self):
        graph_file = 'built_graph/yolov2-tiny-voc.graph'
        meta_file = 'built_graph/yolov2-tiny-voc.meta'
        video_in_path = '/datos/object-detection/videos_to_test/manana.avi'
        threshold = 0.25
        print('Start configuration')
        self.video = cv2.VideoCapture()
        self.video.open(video_in_path)
        self.meta = yolo_utils.get_meta(meta_file)
        self.meta['thresh'] = threshold
        self.dev = movidus_utils.get_mvnc_device()
        self.graph, self.input_fifo, self.output_fifo = movidus_utils.load_graph(
            self.dev, graph_file)
        print('Finished configuration')

    def __del__(self):
        self.video.release()
        self.input_fifo.destroy()
        self.output_fifo.destroy()
        self.graph.destroy()
        self.dev.close()
        self.dev.destroy()

    def get_frame(self):
        print('New frame')
        success, frame = self.video.read() 
        frame_orig = np.copy(frame)
        img_orig_dimensions = frame_orig.shape
        frame = yolo_utils.pre_proc_img(frame, self.meta)
        self.graph.queue_inference_with_fifo_elem(
            self.input_fifo, 
            self.output_fifo,
            frame, 'user object')
        output, user_obj = self.output_fifo.read_elem()
        y_out = np.reshape(output, (13, 13, 125))
        y_out = np.squeeze(y_out)
        boxes = yolo_utils.procces_out(y_out, self.meta, img_orig_dimensions)
        yolo_utils.add_bb_to_img(frame_orig, boxes)
        ret, jpeg = cv2.imencode('.jpg', frame_orig)
        return jpeg.tobytes()
