import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.utils import save_image

# path with segmented images
path_train = 'segmented_files/train'
path_test = 'segmented_files/test'

# list of paths to iterate over
paths = [path_train, path_test]

# iterate over xml files
for i, path in enumerate(paths):
    for filename in os.listdir(path):
        if os.path.splitext(filename)[-1] == '.jpg':
            
            img = cv2.imread(os.path.join(path, filename))
            
            # switch to correct rgb channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # convert to tensor
            convert_tensor = T.ToTensor()
            img = convert_tensor(img)

            # create random rotation, keep complete image with expand = True
            rotater = T.RandomRotation(degrees=(0, 359), expand=True)
            rotated_imgs = [rotater(img) for _ in range(9)]

            # save rotated images
            for j, img_rot in enumerate(rotated_imgs):
                filename_rot = filename[:-4] + f"_rot_{j}" + ".jpg"
                save_image(img_rot, os.path.join(path, 'augmented', filename_rot))




