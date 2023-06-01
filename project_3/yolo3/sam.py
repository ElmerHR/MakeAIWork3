import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

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


image = cv2.imread('/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/apple_disease_classification/yolov3/train/Rot_Apple_64.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
coordinates = [100,100,200,200]


# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()
# plt.close()



sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam, )

predictor.set_image(image)

input_box = np.array(coordinates)
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
input_point = np.array([[input_point_calc_x1, input_point_calc_y1], [input_point_calc_x2, input_point_calc_y2], [input_point_calc_x3, input_point_calc_y3], [input_point_calc_x4, input_point_calc_y4], [input_point_calc_x5, input_point_calc_y5]])
input_label = np.array([1, 1, 1, 1, 1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# mask_input = logits[np.argmax(scores), :, :]

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
# show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()