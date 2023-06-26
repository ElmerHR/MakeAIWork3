import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import time
import os
import warnings

from mean_and_std_img import get_mean_and_std

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class Resnet18Model:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes = ('normal', 'botch', 'rot', 'scab')
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = self.model.fc.in_features
        # set number of output classes
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))
        self.model = self.model.to(device)
        # create temporary dataloader to find mean and std of the images in the train dataset
        self.batch_size = 8

        
        # make sure the classes are in the same order as the classes in the csv file
        self.best_model_params_path = os.path.join('cnn', 'resnet18_best_model_params.pt')
        


