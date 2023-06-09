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

from cnn.utils.mean_and_std_img import get_mean_and_std

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

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

        transform = transforms.Compose(
            [
                transforms.Resize(128)
            ])

        trainset = CustomImageDataset('preprocessing/annotations_train.csv', '', transform=transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=0)
        mean, std = get_mean_and_std(trainloader)
        print(mean, std)

        transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.Normalize(mean, std)
            ])

        trainset = CustomImageDataset('preprocessing/annotations_train.csv', '', transform)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True)

        testset = CustomImageDataset('preprocessing/annotations_test.csv', '', transform)
        self.testloader = DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False)
        print(get_mean_and_std(self.trainloader))
        # make sure the classes are in the same order as the classes in the csv file
        self.best_model_params_path = os.path.join('cnn', 'resnet18_best_model_params.pt')
        

    def train(self):

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(self.model.parameters(), lr=0.0001)

        # Decay LR by a factor of 0.1 every 2 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.02)

        # saving training checkpoints
        torch.save(self.model.state_dict(), self.best_model_params_path)
        best_acc = 0.0
        train_losses = []
        test_losses = []
        test_accuracy = []
        epochs = 10

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            self.model.train()
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()

                # print statistics
                running_loss += loss.item()
                # if i % 40 == 39:    # print every 40 mini-batches
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 40:.3f}')
                #     running_loss = 0.0
            
            train_losses.append(running_loss/len(self.trainloader))
            exp_lr_scheduler.step()

            correct = 0
            total = 0
            running_loss = 0.0
            self.model.eval()
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    # calculate outputs by running images through the network
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                test_losses.append(running_loss/len(self.testloader))
                print(f'[{epoch + 1}] train loss: {running_loss / len(self.trainloader):.3f} test loss: {running_loss / len(self.testloader):.3f}')

            epoch_acc = 100 * correct // total
            print(f'Accuracy of the network on the 116 test images: {epoch_acc} %')
            test_accuracy.append(epoch_acc)
            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.model.state_dict(), self.best_model_params_path)

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
    
    def test(self):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}
        # load best performing model
        self.model.load_state_dict(torch.load(self.best_model_params_path))
        self.model.eval()
        # again no gradients needed
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        print("Best performing model:")
        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
