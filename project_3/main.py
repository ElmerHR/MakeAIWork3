# Import the libraries
import os
import warnings
import argparse
import torch

from torchvision.io import read_image
import torchvision.transforms.functional as F
from sentence_transformers import SentenceTransformer, util

from preprocessing.sam import Segmentation
from preprocessing.augmentation import Augmentation
from preprocessing.annotations import Annotation
from cnn.cnn import CNNModel
from cnn.resnet18 import Resnet18Model

warnings.filterwarnings("ignore")

# constants
NORMALIZE_MEAN = (87.2653, 61.1481, 37.2793)
NORMALIZE_STD = (92.8159, 72.6130, 49.0646)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        lr_list = [0.001, 0.0005, 0.0001, 0.00008]

        cnn_model = CNNModel()
        for lr in lr_list:
            print(f'Training CNN model...\nlr: {lr}')
            cnn_model.train(2, lr)
            print('Testing CNN model...')
            cnn_model.test()
        
    
    if args.train_resnet18:
        print('Training Resnet18 model...')
        resnet18_model = Resnet18Model()
        resnet18_model.train()
        print('Testing Resnet18 model...')
        resnet18_model.test()
    
    if args.aql:
        predictor = None
        # best model paths
        if args.aql and (args.train_cnn or args.train_resnet18):
            print('Aql after training network')
            if args.train_cnn:
                predictor = CNNModel(load_dataset=False)
                predictor.model.to(device)
                predictor.model.load_state_dict(torch.load(predictor.best_model_params_path))
            else:
                predictor = Resnet18Model()
                predictor.model.to(device)
                predictor.model.load_state_dict(torch.load(predictor.best_model_params_path))
            
        else:
            while True:
                model = input("Which trained model do you want to use for making predictions? [cnn or resnet18]: ")

                if model == 'cnn':
                    print("AQL with cnn")
                    predictor = CNNModel(load_dataset=False)
                    predictor.model.to(device)
                    predictor.model.load_state_dict(torch.load(predictor.best_model_params_path))
                    break
                elif model == 'resnet18':
                    print("AQL with resnet18")
                    predictor = Resnet18Model()
                    predictor.model.to(device)
                    predictor.model.load_state_dict(torch.load(predictor.best_model_params_path))
                    break
                else:
                    continue
        
        # tell torch to run in eval mode
        predictor.model.eval()
        # dict to hold predictions
        predictions = {classname: 0 for classname in predictor.classes}
        
        # don't calculate gradients
        with torch.no_grad():
            for i, filename in enumerate(os.listdir('sample_apples')):
                if os.path.splitext(filename)[-1] == '.jpg':
                    image = read_image(os.path.join('sample_apples', filename)).float()
                    # removing alpha channel from image (4th channel)
                    image = image[:3, :, :]
                    # Resize image to 128 x 128
                    image = F.resize(image, (128, 128))
                    # Normalize image
                    image = F.normalize(image, NORMALIZE_MEAN, NORMALIZE_STD)
                    # unsqeeze to create batchsize of 1 (prevents flatten error)
                    image = image.unsqueeze(0)
                    # make prediction
                    output = predictor.model(image.to(device))
                    prediction = torch.argmax(output, dim=1)
                    predictions[predictor.classes[prediction]] += 1
        
        # determine number of bad apples
        bad_apples = sum(predictions.values()) - predictions['normal']
        # aql cutoff points (these are the number of acceptable bad apples)
        aql_class_1 = 0
        aql_class_2 = 3
        aql_class_3 = 7
        aql_class_4 = 8

        # determine the class of the batch of apples
        if bad_apples >= aql_class_4:
            approved = False
            response = "Deze batch van 500 appels is afgekeurd!"
            intended_use = ""
        elif bad_apples <= aql_class_1:
            approved = True
            intended_use = " to supermarkets"
        elif bad_apples <= aql_class_2:
            approved = True
            intended_use = " to apple sauce manufacturers"
        elif bad_apples <= aql_class_3:
            approved = True
            intended_use = " to apple syrup manufacturers"
        
        model = SentenceTransformer('all-MiniLM-L12-v2')

        embeddings = [
                        f"This batch is {'not' if not approved else ''} approved with {'only' if not approved else ''} {20-bad_apples} healthy apples",
                        f"The result of the inspection is that the batch of apples is {'not ' if approved else ''}rejected with {'only ' if approved else ''}{bad_apples} bad apples",
                        f"These apples will {'not ' if not approved else ''}be sold{intended_use}."
                    ]
        print(embeddings)
        query_embedding = model.encode(input('Ask a question about the batch of apples?\n'))
        passage_embedding = model.encode(embeddings)

        answer = util.dot_score(query_embedding, passage_embedding)
        print(embeddings[answer.argmax()])
       
    
if __name__ == "__main__":
    main()