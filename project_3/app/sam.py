import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import sys
from random import randint

from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import tqdm

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

# set checkpoint and model type
# h will run on 8GB GPU card
# sam_checkpoint = "sam_vit_h_4b8939.pth"
# b will run on 4GB GPU card
sam_checkpoint = "sam_vit_b_01ec64.pth"
# model_type = "vit_h"
model_type = "vit_b"

# use GPU
device = "cuda"

# init model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# init Predictor
predictor = SamPredictor(sam)

def segment(img):
    img = np.array(img)
    # print(open_cv_image.shape)
    # # Convert RGB to BGR 
    # open_cv_image = open_cv_image[:, :, ::-1].copy() 
    # print(open_cv_image.shape)
    # xmax,ymax = open_cv_image.shape
    xmax,ymax, _ = img.shape

    # # change from BGR to RGB
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # # set image for predictor
    predictor.set_image(img)

    # # instead of bounding boxes, I use points in the bounding box to increase performance
    input_point_calc_x1 = xmax/2
    input_point_calc_y1 = ymax/2
    input_point_calc_x2 = xmax/6
    input_point_calc_y2 = ymax/6
    input_point_calc_x3 = xmax/6*5
    input_point_calc_y3 = ymax/6
    input_point_calc_x4 = xmax/4
    input_point_calc_y4 = ymax/4*3
    input_point_calc_x5 = xmax/4*3
    input_point_calc_y5 = ymax/4*3

    # save coordinates to input points array
    input_point = np.array([[input_point_calc_x1, input_point_calc_y1],
                            [input_point_calc_x2, input_point_calc_y2],
                            [input_point_calc_x3, input_point_calc_y3],
                            [input_point_calc_x4, input_point_calc_y4],
                            [input_point_calc_x5, input_point_calc_y5]])
    # label = 1 means this is where the area of interest is
    input_label = np.array([1, 1, 1, 1, 1])

    # predict masks
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # print(masks[0].shape)
    # print(masks[0].shape[-2:])
    
    # print(masks[0].reshape(h, w, 1))
    # print(type(masks[0].reshape(h, w, 1)))
    h, w = masks[0].shape
    # create mask, cv2.bitwise_and only accepts uint8 arrays
    mask = np.array(masks[0].reshape(h, w, 1), np.uint8)
    # print(type(mask))
    #  Bitwise-AND mask and original image --> this will only save the underlying image where the mask is
    result = cv2.bitwise_and(img,img, mask= mask)

    print(result.shape)

    img_crop = crop(result)

    # get old size shape
    old_size = img_crop.shape[:2] # old_size is in (height, width) format

    # find longest edge
    max_size = max(old_size)

    # calculate delta between max_size and old size edge
    delta_w = max_size - old_size[1]
    delta_h = max_size - old_size[0]

    # calculate image placement
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    # place border around image in color black
    color = [0, 0, 0]
    img_crop_sq = cv2.copyMakeBorder(img_crop, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    data = np.array(img_crop_sq)   # "data" is a height x width x 4 numpy array
    red, green, blue = data.T # Temporarily unpack the bands for readability

    # Replace black with random noise
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    shape = data[...][black_areas.T].shape
    print(shape)
    Z = np.random.rand(shape[0], shape[1]) * 255
    data[...][black_areas.T] = Z

    return data
