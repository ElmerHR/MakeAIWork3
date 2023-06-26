#--------------------------------------------------

import time
import numpy as np
import os
import streamlit as st
import torch
import torchvision.transforms.functional as F
from PIL import Image
from streamlit_cropper import st_cropper

from sam import segment
from resnet18 import Resnet18Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# constants
NORMALIZE_MEAN = (87.2653, 61.1481, 37.2793)
NORMALIZE_STD = (92.8159, 72.6130, 49.0646)

#--------------------------------------------------

def restart():
    
    # Reset the page indicator
    st.session_state.page = 'upload'

    # Clear the image batch
    st.session_state.batch = None
    st.session_state.cropped_batch = None
    st.session_state.batch_length = None
    st.session_state.batch_idx = None
    st.session_state.segmented_batch = []

#--------------------------------------------------

def toCropper():
    st.session_state.page = 'cropper'
    st.session_state.batch_idx = 0

#--------------------------------------------------

def nextImage():
    st.session_state.batch_idx += 1

#--------------------------------------------------

def previousImage():
    st.session_state.batch_idx -= 1

#--------------------------------------------------

def toSegmenter():
    st.session_state.page = 'segmenter'

#--------------------------------------------------

def upload():

    with page.container():

        st.title("Upload 20 apple samples for inspection")
        # Store images
        st.session_state.batch = st.file_uploader("", type=None, accept_multiple_files=True)
        if len(st.session_state.batch) == 20:

            st.session_state.batch_length = len(st.session_state.batch)

            if st.session_state.batch:
                st.button("Process images", on_click=toCropper)
        elif len(st.session_state.batch) > 0:
            st.warning('Please upload exactly 20 images.', icon="⚠️")

#--------------------------------------------------

def cropper():

    with page.container():

        st.title("Cropping images")
        st.write(f"Image {st.session_state.batch_idx + 1} of {len(st.session_state.batch)}.")
        
        box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
        
        img_file = st.session_state.batch[st.session_state.batch_idx]
        
        # open image and remove alpha channel, which we don't need
        img = Image.open(img_file).convert('RGB') 
        
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=True, box_color=box_color,
                                    aspect_ratio=(1,1))
        
        st.session_state.cropped_batch[st.session_state.batch_idx] = cropped_img
        print(st.session_state.cropped_batch)
        col1, col2 = st.columns([.3,1])
        # display nav buttons
        if st.session_state.batch_idx + 1 != st.session_state.batch_length:
            col2.button("Next image", on_click=nextImage)
        if st.session_state.batch_idx > 0:
            col1.button("Previous image", on_click=previousImage)
        if st.session_state.batch_idx + 1 == st.session_state.batch_length:
            col2.button("Save all images", on_click=toSegmenter)
        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)
        st.button("Back to home", on_click=restart)

#--------------------------------------------------

def aql(bad_apples):
    with page.container():

        st.title("Batch inspection")

        # aql cutoff points (these are the number of acceptable bad apples)
        aql_class_1 = 0
        aql_class_2 = 3
        aql_class_3 = 7
        aql_class_4 = 8

        # determine the class of the batch of apples
        if bad_apples >= aql_class_4:
            st.write("This batch of 500 apples is rejected!")
        elif bad_apples <= aql_class_1:
           st.write("This batch of 500 apples is of supermarket quality!")
        elif bad_apples <= aql_class_2:
           st.write("This batch of 500 apples will be processed to apple sauce!")
        elif bad_apples <= aql_class_3:
           st.write("This batch of 500 apples will be processed to apple syrup!")

#--------------------------------------------------

def classify():

    with page.container():

        st.title("Classifying apples...")

        with st.spinner('Please wait for classifier to finish...'):
            # load resnet model and weights
            predictor = Resnet18Model()
            predictor.model.to(device)
            predictor.model.load_state_dict(torch.load(os.path.join('../cnn', 'resnet18_best_model_params.pt')))
            # tell torch to run in eval mode
            predictor.model.eval()
            # dict to hold predictions
            predictions = {classname: 0 for classname in predictor.classes}
            
            # don't calculate gradients
            with torch.no_grad():
                for img in enumerate(st.session_state.segmented_batch):
                    
                    image = torch.tensor(img[1]).float()
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

        st.success('Succesfully classified all apples!')

        aql(bad_apples)
        

#--------------------------------------------------

def segmenter():

    with page.container():

        st.title("Removing background")
        # progress bar showing progress of removing background from images
        progress_text = "Background removal in progress (0%). Please wait..."
        progress_bar = st.progress(0, text=progress_text)

        for i, img in enumerate(st.session_state.cropped_batch.values()):
            # segment anything function
            segmented_image = segment(img)
            # add image to segmented batch list for later use
            st.session_state.segmented_batch.append(segmented_image)
            percent_complete = (i+1)/len(st.session_state.cropped_batch)
            progress_text = f"Background removal in progress ({percent_complete*100:.0f}%). Please wait..."
            progress_bar.progress(percent_complete, text=progress_text)
        
        # remove gpu cache from gpu card
        torch.cuda.empty_cache()

    # classify images
    classify()


#--------------------------------------------------

# Note: Steamlit will run the code below
#       at the start and after button clicks

#--------------------------------------------------

# Only enters when the page indicator is not defined (at startup)
if "page" not in st.session_state:

    # Define the page indicator
    st.session_state.page = 'upload'

    # To store image batch
    st.session_state.batch = None
    st.session_state.cropped_batch = {}
    st.session_state.batch_length = None
    st.session_state.batch_idx = None
    st.session_state.segmented_batch = []


#--------------------------------------------------

# Start with empty page
page = st.empty()

#--------------------------------------------------

# The flow of our app
if st.session_state.page == 'upload':

    upload()

elif st.session_state.page == 'cropper':

    cropper()

elif st.session_state.page == 'segmenter':

    segmenter()

#--------------------------------------------------