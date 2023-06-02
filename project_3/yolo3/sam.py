import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

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
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# set checkpoint and model type
sam_checkpoint = "sam_vit_h_4b8939.pth"
# sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_h"

# use GPU
device = "cuda"

# init model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# init Predictor
predictor = SamPredictor(sam)

# path to xml and image files
path_train = '/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/yolo3/renamed_files/train'
path_test = '/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/yolo3/renamed_files/test'

# path to save segmented images to
segmented_path_train = '/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/yolo3/segmented_files/train'
segmented_path_test = '/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/yolo3/segmented_files/test'

# list of paths to iterate over
paths = [path_train, path_test]
segmented_paths = [segmented_path_train, segmented_path_test]

# iterate over xml files
for i, path in enumerate(paths):
    for filename in os.listdir(path)[:5]:
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
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
            # create figure and save to file
            plt.figure(figsize=(10, 10))
            plt.imshow(result)
            plt.axis('off')
            # plt.show()
            plt.savefig(os.path.join(segmented_paths[i], image_filename))
            plt.close()

