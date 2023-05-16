import io
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from skimage.measure import block_reduce

class ImageFilter(object):
    
    def __init__(self, kernel):
        self.imgKernel = kernel
    
    def convolve(self, imgTensor):
        imgTensorRGB = imgTensor.copy() 
        outputImgRGB = np.empty_like(imgTensorRGB)

        for dim in range(imgTensorRGB.shape[-1]):  # loop over rgb channels
            outputImgRGB[:, :, dim] = sp.signal.convolve2d (
                imgTensorRGB[:, :, dim], self.imgKernel, mode="same", boundary="symm"
            )
        
        return outputImgRGB
    
    def down_sample(self, imgTensor):
        return block_reduce(imgTensor, block_size = (2, 2, 1), func=np.max)
