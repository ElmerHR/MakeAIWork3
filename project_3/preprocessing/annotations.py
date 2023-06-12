import os
import csv
from tqdm import tqdm

class Annotation():

    def __init__(self):
        pass

    def annotate(self):
        # path with segmented images
        path_train = 'preprocessing/segmented_files/train'
        path_test = 'preprocessing/segmented_files/test'

        # list of paths to iterate over
        paths = [path_train, path_test]

        # classes to idc
        classes_idx = {'normal': 0, 'botch': 1, 'rot': 2, 'scab': 3}

        # lists to store data to write to csv file
        to_csv_train = []
        to_csv_test = []

        # list of csv list to iterate over
        csvs = [to_csv_train, to_csv_test]

        # iterate over all images in both paths
        for i, path in enumerate(paths):
            for filename in tqdm(os.listdir(path)):
                if os.path.splitext(filename)[-1] == '.jpg':
                    # get label name from filename
                    label = filename.split('_', 1)[0]
                    # find label idx belonging to label text
                    label_idx = classes_idx.get(label)
                    # get full path to stored image
                    path_img = os.path.join(path, filename)
                    # print(label, label_idx, path_img)
                    # append filepath and class idx to csv list
                    csvs[i].append([path_img, label_idx])
                if filename in ['dr', 'gn', 'hf', 'lg', 'sp', 'rotated']:
                    for filename_augmented in tqdm(os.listdir(os.path.join(path, filename))):
                        # get label name from filename
                        label = filename_augmented.split('_', 1)[0]
                        # find label idx belonging to label text
                        label_idx = classes_idx.get(label)
                        # get full path to stored image
                        path_img = os.path.join(path, filename, filename_augmented)
                        # print(label, label_idx, path_img)
                        # append filepath and class idx to csv list
                        csvs[i].append([path_img, label_idx])

        # iterate over csvs list
        for i, csv_list in enumerate(csvs):
            # if == 0, we create the train annotations, otherwise test
            if i == 0:
                annotations = 'train'
            else:
                annotations = 'test'
            # save list to csv file
            with open(f"preprocessing/annotations_{annotations}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(csv_list)
        print("Finished creating annotations file...")