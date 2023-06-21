#--------------------------------------------------

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from sam import segment
import time

#--------------------------------------------------

def restart():
    
    # Reset the page indicator
    st.session_state.page = 'upload'

    # Clear the image batch
    st.session_state.batch = None
    st.session_state.cropped_batch = None
    st.session_state.batch_length = None
    st.session_state.batch_idx = None

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

        st.title("Upload an image batch")
        # Store images
        st.session_state.batch = st.file_uploader("", type=None, accept_multiple_files=True)
        st.session_state.batch_length = len(st.session_state.batch)
        if st.session_state.batch:
            st.button("Process images", on_click=toCropper)

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
        # display nav buttons
        if st.session_state.batch_idx + 1 != st.session_state.batch_length:
            st.button("Next image", on_click=nextImage)
        if st.session_state.batch_idx > 0:
            st.button("Previous image", on_click=previousImage)
        if st.session_state.batch_idx + 1 == st.session_state.batch_length:
            st.button("Save all images", on_click=toSegmenter)
        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)
        st.button("Back to home", on_click=restart)


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
            st.image(segmented_image)
            time.sleep(2)
            percent_complete = (i+1)/len(st.session_state.cropped_batch)
            progress_text = f"Background removal in progress ({percent_complete*100:.0f}%). Please wait..."
            progress_bar.progress(percent_complete, text=progress_text)


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