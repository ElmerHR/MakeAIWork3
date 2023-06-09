import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
from torchvision.utils import save_image
from skimage.util import random_noise
from tqdm import tqdm


# path with segmented images
path_train = 'preprocessing/segmented_files/train/noise'
path_test = 'preprocessing/segmented_files/test/noise'

# list of paths to iterate over
paths = [path_train]

class Augmentation:

    def __init__(self):
        pass

    def augment(self):
        print("Augmenting images (Salt & Pepper noise, Gaussian nose, Lighten,\nDarken and horizontal flip)...")
        # iterate over all files
        for i, path in enumerate(paths):
            for filename in tqdm(os.listdir(path)):
                if os.path.splitext(filename)[-1] == '.png':
                    pass
                    # img = cv2.imread(os.path.join(path, filename))
                    
                    # # switch to correct rgb channel
                    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # # salt pepper noise
                    # img_sp = random_noise(img_rgb, mode='s&p', amount=0.1)

                    # # gaussian noise
                    # img_gn = random_noise(img_rgb, mode='gaussian', mean=0, var=0.05, clip=True)

                    # # brighten/darken image
                    # convert_tensor = T.ToTensor()
                    # img_t = convert_tensor(img_rgb)
                    # img_lg_t = T.functional.adjust_brightness(img_t, 1.5)
                    # img_dr_t = T.functional.adjust_brightness(img_t, 0.9)

                    # # horizontal_flip
                    # img_hf_t = T.functional.hflip(img_t)

                    # # save images
                    
                    # img_sp_t = convert_tensor(img_sp)
                    # img_gn_t = convert_tensor(img_gn)
                    
                    # filename_sp = filename[:-4] + "_sp" + ".png"
                    # filename_gn = filename[:-4] + "_gn" + ".png"
                    # filename_lg = filename[:-4] + "_lg" + ".png"
                    # filename_dr = filename[:-4] + "_dr" + ".png"
                    # filename_hf = filename[:-4] + "_hf" + ".png"

                    # save_image(img_sp_t, os.path.join(path, 'sp', filename_sp))
                    # save_image(img_gn_t, os.path.join(path, 'gn', filename_gn))
                    # save_image(img_lg_t, os.path.join(path, 'lg', filename_lg))
                    # save_image(img_dr_t, os.path.join(path, 'dr', filename_dr))
                    # save_image(img_hf_t, os.path.join(path, 'hf', filename_hf))
        
        print("Augmenting images, rotation of all augmented images...")
        # iterate over all images in both paths
        for i, path in enumerate(paths):
            for filename in tqdm(os.listdir(path)):
                if os.path.splitext(filename)[-1] == '.png':
                    img = cv2.imread(os.path.join(path, filename))
                    # switch to correct rgb channel
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # degrees of rotation
                    degrees = [60, 120, 180, 240, 300]
                    # list to store rotated images
                    rotated_imgs = []
                    for degree in degrees:
                        img_rot = scipy.ndimage.rotate(img_rgb, degree)
                        red, green, blue = img_rot.T # Temporarily unpack the bands for readability

                        # Replace black with random noise
                        black_areas = (red == 0) & (blue == 0) & (green == 0)
                        shape = img_rot[...][black_areas.T].shape
                        Z = np.random.rand(shape[0], shape[1]) * 255
                        img_rot[...][black_areas.T] = Z
                        # convert img array back to BGR format
                        img_bgr = cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR)
                        filename_rot = filename[:-4] + f"_rot_{degree}" + ".png"
                        cv2.imwrite(os.path.join(path, 'rotated', filename_rot), img_bgr)


                if filename in ['dr', 'gn', 'hf', 'lg', 'sp']:
                    for filename_augmented in os.listdir(os.path.join(path, filename)):
                        if os.path.splitext(filename_augmented)[-1] == '.png':
                            img = cv2.imread(os.path.join(path, filename, filename_augmented))
                            # switch to correct rgb channel
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # degrees of rotation
                            degrees = [60, 120, 180, 240, 300]
                            # list to store rotated images
                            for degree in degrees:
                                img_rot = scipy.ndimage.rotate(img_rgb, degree)
                                red, green, blue = img_rot.T # Temporarily unpack the bands for readability

                                # Replace black with random noise
                                black_areas = (red == 0) & (blue == 0) & (green == 0)
                                shape = img_rot[...][black_areas.T].shape
                                Z = np.random.rand(shape[0], shape[1]) * 255
                                img_rot[...][black_areas.T] = Z
                                # convert img array back to BGR format
                                img_bgr = cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR)
                                filename_rot = filename_augmented[:-4] + f"_rot_{degree}" + ".png"
                                cv2.imwrite(os.path.join(path, 'rotated', filename_rot), img_bgr)

        print("Finished augmenting images...")
