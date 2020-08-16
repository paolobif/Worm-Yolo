import os
import cv2
import csv
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from yolov3_core import YoloModelLatest
from yolov3_core import ImageInLine
from yolov3_core import ImageProcessor
from yolov3_core import *

def load_images_to_array(im_dir):
    #data_ = defaultdict()
    data__ = []
    image_names = os.listdir(im_dir)
    for name in image_names:
        img0 = cv2.imread(f"{im_dir}/{name}")
        data__.append(img0)
    return data__, image_names


def draw_on_im(img, x1, y1, x2, y2, conf, col, text=None):
    center_x = (x2-x1)/2 + x1
    center_y = (y2-y1)/2 + y1

    cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
    if text is not None:
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)


# creates pandas df for easy csv saving.
def pd_for_csv(outputs, img_name = "name"):
    csv_outputs = []
    for output in outputs:
        x1, y1, x2, y2, *_ = output
        w = abs(x2-x1)
        h = abs(y2-y1)
        csv_outputs.append([img_name, x1.tolist(), y1.tolist(), w.tolist(), h.tolist()]) # ideally change to list earlier bc now outputs is a mix of tensors and lists....
    out_df = pd.DataFrame(csv_outputs)
    return out_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="cfg/yolov3-spp-1cls.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="416_8_9/best.pt", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="cfg/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold required to qualify as detected def 0.4")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold def 0.001")
    parser.add_argument("--no_gpu", default=True, help="cuda enabled gpu")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension of first layer of yolo")
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')

    # select input and output location also output options
    parser.add_argument('--out_path', type=str, default='output/custom')
    parser.add_argument("--data_path", type=str, default='data/samples', help="path the video file or to the directory with stack of images")
    parser.add_argument("--video", action='store_true', help="save the labled images as a video")
    parser.add_argument("--csv", action='store_true', help="save the bounding box data into a csv in the out directory")
    parser.add_argument("--img", action='store_true', help="store as image")
    opt = parser.parse_args()

    # create settings dictionaryprint(input_img.shape)
    settings = {'model_def': opt.model_def,
                'weights_path': opt.weights_path,
                'class_path': opt.class_path,
                'img_size': opt.img_size,
                'iou_thres': opt.iou_thres,
                'no_gpu': opt.no_gpu,
                'conf_thres': opt.conf_thres,
                'batch_size': opt.batch_size,
                'augment': opt.augment,
                'classes': opt.classes}

    # number depends on which weights being used.
    SLICE_SIZE = 480 #480 is a temp fix to solve worms not recognized towards bottom. must add padding in future. Ideal is 416

    # determine input type
    INPUT_VIDEO = bool
    if ".avi" in opt.data_path:
        INPUT_VIDEO = True
        print("--Input being processed as a video--")
    else:
        INPUT_VIDEO = False
        print("--Input is stack of images--")

    def save_img(img, outputs, file_path):
        pass


    # fork video out vs images out as the output currently not setup fror images to video
    OUT_VIDEO = opt.video
    if OUT_VIDEO == True:
        if INPUT_VIDEO: out_video_path = f"{opt.out_path}/{os.path.basename(opt.data_path).strip('.avi')}_anotated.avi"
        if not INPUT_VIDEO: out_video_path = f"{opt.out_path}/{os.path.basename(opt.data_path)}_anotated.avi"

        if os.path.exists(out_video_path):
            n = 1
            while os.path.exists(out_video_path):
                out_video_path = f"{out_video_path.strip('.avi')}_anotated{n}.avi"
                n += 1
        # set up video capture and write
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_video_path, fourcc, 10, (1920, 1080), True) # currently set to 10fps


    # load yolov3 model and start processing
    Yolo = YoloModelLatest(settings)
    # start parsing and processing
    if INPUT_VIDEO == True:
        vid = cv2.VideoCapture(opt.data_path)
        while (1):
            ret, frame = vid.read()
            frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)

            frame_obj = ImageProcessor(frame, out_size=SLICE_SIZE)
            input_dict = frame_obj.image_slices
            outputs = Yolo.pass_model(input_dict)

    elif INPUT_VIDEO == False:
        stack, file_names = load_images_to_array(opt.data_path)
        for frame, name in zip(stack, file_names):
            frame_obj = ImageProcessor(frame, out_size=SLICE_SIZE)
            input_dict = frame_obj.image_slices
            outputs = Yolo.pass_model(input_dict)

            if opt.csv == True:
                raw_name, extension = name.split(".")
                csv_df = pd_for_csv(outputs, img_name=name)
                csv_df.to_csv(f"{opt.out_path}/{raw_name}_NN.csv", header=None, index=None)

            if opt.img == True:
                for output in outputs:
                    x1, y1, x2, y2, conf, cls_conf = output
                    draw_on_im(frame, x1, y1, x2, y2, conf, (100,255,0), text="Worm")
                cv2.imwrite(f"{opt.out_path}/{name}_anotated.png", frame)
