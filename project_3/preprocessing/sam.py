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

sys.path.append("..")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

class Segmentation:

    def __init__(self):
        pass

    def segment(self):
        # set checkpoint and model type
        # h will run on 8GB GPU card
        # sam_checkpoint = "sam_vit_h_4b8939.pth"
        # b will run on 4GB GPU card
        sam_checkpoint = "preprocessing/sam_vit_b_01ec64.pth"
        # model_type = "vit_h"
        model_type = "vit_b"

        # use GPU
        device = "cuda"

        # init model
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        # init Predictor
        predictor = SamPredictor(sam)

        # path to xml and image files
        path_train = 'preprocessing/renamed_files/train'
        path_test = 'preprocessing/renamed_files/test'

        # path to save segmented images to
        segmented_path_train = 'preprocessing/segmented_files/train'
        segmented_path_test = 'preprocessing/segmented_files/test'

        # list of paths to iterate over
        paths = [path_train, path_test]
        segmented_paths = [segmented_path_train, segmented_path_test]

        # iterate over xml files
        for i, path in enumerate(paths):
            for filename in os.listdir(path):
                if os.path.splitext(filename)[-1] == '.xml':
                    tree = ET.parse(os.path.join(path, filename))
                    root = tree.getroot()
                    # get image filename
                    image_filename = root[1].text
                    # get image path
                    image_path = root[2].text
                    # load image
                    image = cv2.imread(image_path)
                    # retrieve bounding box coordinates
                    xmin = int(root[6][4][0].text)
                    ymin = int(root[6][4][1].text)
                    xmax = int(root[6][4][2].text)
                    ymax = int(root[6][4][3].text)

                    coordinates = [xmin,ymin,xmax,ymax]

                    # change from BGR to RGB
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # set image for predictor
                    predictor.set_image(image)

                    # instead of bounding boxes, I use points in the bounding box to increase performance
                    input_point_calc_x1 = coordinates[0] + (coordinates[2]-coordinates[0])/2
                    input_point_calc_y1 = coordinates[1] + (coordinates[3]-coordinates[1])/2
                    input_point_calc_x2 = coordinates[0] + (coordinates[2]-coordinates[0])/6
                    input_point_calc_y2 = coordinates[1] + (coordinates[3]-coordinates[1])/6
                    input_point_calc_x3 = coordinates[0] + (coordinates[2]-coordinates[0])/6*5
                    input_point_calc_y3 = coordinates[1] + (coordinates[3]-coordinates[1])/6
                    input_point_calc_x4 = coordinates[0] + (coordinates[2]-coordinates[0])/4
                    input_point_calc_y4 = coordinates[1] + (coordinates[3]-coordinates[1])/4*3
                    input_point_calc_x5 = coordinates[0] + (coordinates[2]-coordinates[0])/4*3
                    input_point_calc_y5 = coordinates[1] + (coordinates[3]-coordinates[1])/4*3

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
                    result = cv2.bitwise_and(image,image, mask= mask)

                    # print(result.shape)

                    img_crop = crop(result)
                    # print(img_crop.shape)

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
                    # img_crop_noise = Image.fromarray(data)
                    # cv2.imshow("image", img_crop_sq)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # save image to file
                    cv2.imwrite(os.path.join(segmented_paths[i], 'noise', image_filename[:-4]+'.png'), data)
        print("Finalized segmentation of images...")
