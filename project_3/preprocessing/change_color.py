from PIL import Image
import numpy as np
import random 

im = Image.open('/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/preprocessing/segmented_files/train/botch_apple_15.jpg')
# im = im.convert('RGB')

data = np.array(im)   # "data" is a height x width x 4 numpy array
red, green, blue = data.T # Temporarily unpack the bands for readability

# Replace black with random noise
black_areas = (red == 0) & (blue == 0) & (green == 0)
shape = data[...][black_areas.T].shape
print(shape)
Z = np.random.rand(shape[0], shape[1]) * 255
data[...][black_areas.T] = Z

im2 = Image.fromarray(data)
im2.save('/home/elmer/dev_debian/working/python3/MakeAIWork3/project_3/preprocessing/segmented_files/train/botch_apple_15_noise.png')