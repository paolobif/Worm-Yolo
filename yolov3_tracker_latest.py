import os
import cv2
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict
from yolov3_core import YoloModelLatest
from yolov3_core import ImageInLine
from yolov3_core import ImageProcessor
from yolov3_core import *

VIDEO = False
video_directory = ''

images_directory = 'data/test_images'

def load_images_to_array(im_dir):
    #data_ = defaultdict()
    data__ = []
    image_names = os.listdir(im_dir)
    for name in image_names:
        img0 = cv2.imread(f"{im_dir}/{name}")
        data__.append(img0)
    return data__


def draw_on_im(img, x1, y1, x2, y2, conf, col):
    center_x = (x2-x1)/2 + x1
    center_y = (y2-y1)/2 + y1

    cv2.rectangle(
        img,
        (x1,y1),(x2,y2),
        col, 2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="cfg/yolov3-spp-1cls.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/best.pt", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="cfg/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold required to qualify as detected def 0.4")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold def 0.001")
    parser.add_argument("--no_gpu", default=True, help="cuda enabled gpu")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension of first layer of yolo")
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')

    parser.add_argument('--out_path', type=str, default='output/custom/')
    parser.add_argument("--data_path", type=str, default='data/samples', help="path the video file or to the directory with stack of images")
    parser.add_argument("--video", action='store_true', help="save the labled images as a video")
    parser.add_argument("--csv", action='store_true', help="save the bounding box data into a csv in the out directory")
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


    # determine input type
    INPUT_VIDEO = bool
    if ".avi" in opt.data_path:
        INPUT_VIDEO = True
        print("\n Input being processed as a video \n")


    # load yolov3 model
    yolo = YoloModelLatest(settings)

    # runs on image stack in folder
    if VIDEO == False:
        frames = load_images_to_array(images_directory)
        x=0
        for frame in frames:
            print(frame.shape)
            img_obj = ImageProcessor(frame, out_size=416)
            input_dict = img_obj.image_slices
            # input is in format [image, position]
            # it breaks on the last frame not sure a good work around

            outputs = yolo.pass_model(input_dict)

            #img_obj.rescale_bboxes(outputs)
            for output in outputs:
                x1, y1, x2, y2, conf, cls_conf = output
                draw_on_im(frame, x1, y1, x2, y2, conf, (100,255,0))


            cv2.imwrite(f"output/custom/{x}.png", frame)
            x += 1
