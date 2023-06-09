import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from skimage.util import random_noise



from utils.mean_and_std_img import get_mean_and_std

import warnings

warnings.filterwarnings("ignore")

# custom Image Dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.pool = nn.MaxPool2d(3, 2)
        
        self.fc1 = nn.Linear(64*3*3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

class SaltPepper(object):
    """Add salt and pepper noise.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, amount=0.05):
        self.amount = amount

    def __call__(self, sample):
        image = sample
        # print(image)
        image = torch.Tensor(random_noise(image.numpy(), mode='salt', amount=self.amount))
        # print(image)
        return image

class GaussianNoise(object):
    """Add salt and pepper noise.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample
        # print(image)
        image = torch.Tensor(random_noise(image.numpy(), mode='gaussian', mean=0, var=0.05, clip=True))
        # print(image)
        return image
    
class AdjustBrightness(object):
    """Add salt and pepper noise.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        image = sample
        # print(image)
        image = torch.Tensor(transforms.functional.adjust_brightness(image, self.factor))
        # print(image)
        return image

net = Net()
summary(net, (3, 128, 128))

# create temporary dataloader to find mean and std of the images in the train dataset
batch_size = 8

transform = transforms.Compose(
    [
        transforms.Resize(128)
    ])

trainset = CustomImageDataset('../preprocessing/annotations_train.csv', '../preprocessing', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
mean, std = get_mean_and_std(trainloader)
print(mean, std)

transform = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.Normalize(mean, std)
    ])

batch_size = 8

trainset = CustomImageDataset('../preprocessing/annotations_train.csv', '../preprocessing', transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = CustomImageDataset('../preprocessing/annotations_test.csv', '../preprocessing', transform)
testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
print(get_mean_and_std(trainloader))
# make sure the classes are in the same order as the classes in the csv file
classes = ('normal', 'botch', 'rot', 'scab')

# functions to show an image
def imshow(img):
    
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    img_norm = (npimg-np.min(npimg))/(np.max(npimg)-np.min(npimg))
    plt.imshow(np.transpose(img_norm, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0008)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.03)
train_losses = []
test_losses = []
test_accuracy = []
epochs = 8
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 40 == 39:    # print every 40 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 40:.3f}')
        #     running_loss = 0.0
    print(f'[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}')
    train_losses.append(running_loss/len(trainloader))
    # exp_lr_scheduler.step()

    correct = 0
    total = 0
    running_loss = 0.0
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_losses.append(running_loss/len(testloader))


    print(f'Accuracy of the network on the 1160 test images: {100 * correct // total} %')
    test_accuracy.append(100 * correct // total)

print('Finished Training')
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.tight_layout()
ax1.plot(range(epochs), train_losses, label='train_loss')
ax1.plot(range(epochs), test_losses, label='test_loss')
ax1.legend()
ax1.set_title('CNN train and test losses')
ax2.plot(range(epochs), test_accuracy)
ax2.set_title('CNN test accuracy')
fig.subplots_adjust(hspace=0.3)
plt.show()

# dataiter = iter(testloader)
# images, labels = next(dataiter)

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# outputs = net(images.to(device))

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                            #   for j in range(4)))



# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
net.eval()
# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')