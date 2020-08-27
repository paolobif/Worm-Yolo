import os
import cv2
import csv
import time
import argparse
import tqdm
import numpy as np
from yolov3_core import *

settings = {
    'model_def': 'cfg/yolov3-spp-1cls.cfg',
    'weights_path': '416_8_9/best.pt',
    'class_path': 'cfg/classes.names',
    'img_size': 608,
    'iou_thres': 0.6,
    'no_gpu': True,
    'conf_thres': 0.3,
    'batch_size': 1,
    'augment': False,
    'classes': ''
    }

# load yolov3 model
Yolo = YoloModelLatest(settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/test_images/335_def', help="path the video file or to the directory with stack of images")
    parser.add_argument('--out_path', type=str, default='output')
    opt = parser.parse_args()

    ## start video writer and designate vid path ##
    out_video_path = f"{opt.out_path}/{os.path.basename(opt.data_path)}_anotated.avi"
    n = 0
    #while os.path.exists(out_video_path):
    #    out_video_path = f"{out_video_path.strip('.avi')}_{n}.avi"
    #    n += 1

    # set up video capture and write
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_video_path, fourcc, 10, (1920, 1080), True) # currently set to 10fps
    print(f"generating video at {out_video_path}")

    # get images
    start_time = time.time()
    file_names = sorted(os.listdir(opt.data_path))
    reverse_file_names = file_names.reverse() #reversed to make TOD calls
    #head_name = os.path.basename(opt.data_path)

    for i, file_name in enumerate(file_names):
        print(file_name)
        frame = cv2.imread(f"{opt.data_path}/{file_name}")
        cv2.imshow("test", frame)
        cv2.waitkey(0)
        frame_obj = ImageProcessor(frame, out_size=480)
        input_dict = frame_obj.image_slices
        outputs = Yolo.pass_model(input_dict)
        print(f"\t Image: {i}/{len(file_names)}")

        for output in outputs:
            x1, y1, x2, y2, conf, cls_conf = output
            draw_on_im(frame, x1, y1, x2, y2, conf, (100,255,0))

        writer.write(frame)

    finish_time = time.time()
    process_time = datetime.timedelta(seconds=finish_time-start_time)
    print(f"{len(file_names)} took {process_time}")
