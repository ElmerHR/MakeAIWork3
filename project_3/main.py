# Import the library
import argparse

from preprocessing.sam import Segmentation
from preprocessing.augmentation import Augmentation
from preprocessing.annotations import Annotation
from cnn.cnn import CNNModel
from cnn.resnet18 import Resnet18Model

def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--annotate', action='store_true')
    # Add mutually exclusive group for custom-cnn and resnet18
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train_cnn', action='store_true')
    group.add_argument('--train_resnet18', action='store_true')

    parser.add_argument('--aql', action='store_true')

    # Parse the argument
    args = parser.parse_args()
    
    if args.segment:
        print("Performing segmentation of raw images...")
        segmentation = Segmentation()
        segmentation.segment()

    if args.augment:
        print('Performing augmentation of segmented images...')
        augmentation = Augmentation()
        augmentation.augment()
    
    if args.annotate:
        print('Creating annotations file...')
        annotation = Annotation()
        annotation.annotate()
    
    if args.train_cnn:
        print('Training CNN model...')
        cnn_model = CNNModel()
        cnn_model.train()
        print('Testing CNN model...')
        cnn_model.test()
    
    if args.train_resnet18:
        print('Training Resnet18 model...')
        resnet18_model = Resnet18Model()
        resnet18_model.train()
        print('Testing Resnet18 model...')
        resnet18_model.test()

    if args.aql and (args.train_cnn or args.train_resnet18):
        print('Aql after training network')
    elif args.aql:
        while True:
            model = input("Which trained model do you want to use for making predictions? [cnn or resnet18]: ")

            if model == 'cnn':
                print("AQL with cnn")
                break
            elif model == 'resnet18':
                print("AQL with resnet18")
                break
            else:
                continue


if __name__ == "__main__":
    main()