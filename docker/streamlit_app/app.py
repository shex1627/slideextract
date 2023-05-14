import streamlit as st
from PIL import Image
from pathlib import Path
from collections import defaultdict
import os
import tempfile
import json
import logging
import pytube
import re
from typing import Dict
import io

from streamlit_util import *
from slideextract.config import VALID_YOUTUBE_CHANNELS, MAX_VIDEO_DURATION, YOUTUBE_VIDEO_DOWNLOAD_PATH, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME
from slideextract.slide_extractors.PDHashOCRFunnel import PDHashOCRFunnel
from slideextract.slide_extractors.PDHashExtractor import PDHashExtractor
from slideextract.slide_extractors.image_to_ppt import create_pptx
from slideextract.ocr.textractOcr import textractOcr

#disable caching for so container doesn't expode
#@st.cache_data(show_spinner="Processing Youtube Video")
def download_youtube_video(link, output_path: str="./youtube_downloads"):
    with st.spinner("Downloading Youtube Video"):
        yt = pytube.YouTube(link)
        title = yt.title.lower().replace(" ","_")
        filename = re.sub('[^A-Za-z0-9_]+', '', title) + ".mp4"

        logger.info(f"resolutions: {yt.streams.filter(progressive=True)}")
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        stream.download(output_path=output_path, filename=filename)
    return os.path.join(output_path, filename)

logger = logging.getLogger("streamlit app")

# define a function ti display image metadata
def display_metadata(metadata):
    for key, value in metadata.items():
        st.write(f"**{key}:** {value}")

# Define the main function that displays the images and their metadata
#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def open_images(files):
    return {file['name']: Image.open(file['path']) for file in files}

def display_images(images_metadata: Dict, image_data: Dict):
    NUM_COLUMNS = 2
    cols = st.columns(NUM_COLUMNS)
    for i, metadata in enumerate(images_metadata):
        with cols[i % NUM_COLUMNS]:
            st.write(f"## {metadata['name']}, Duration {metadata['duration']}")
            #display_metadata(metadata)
            st.image(image_data[metadata['image_index']], use_column_width=True, caption=metadata['duration'], width=150, output_format='PNG')


#@st.cache_data(show_spinner="Extracting Slides, Usually takes 3-6min for a new video")
def run_slide_extraction(mp4_path: str, with_ocr: bool = True):
    with st.spinner('Extracting Slides, Usually takes 3-6min for a new video'):
        if with_ocr:
            #if not slide_extractor.load_slide_extract_output(mp4_path):
            textract_ocr = textractOcr(SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, os.path.basename(mp4_path).split(".")[0])
            slide_extractor = PDHashOCRFunnel(ocr_engine=textract_ocr, duration_threshold=0)
            if not slide_extractor.load_slide_extract_output(mp4_path):
                slide_extractor.extract_slides_from_file(mp4_path)
        else:
            slide_extractor = PDHashExtractor(duration_threshold=0)
            if not slide_extractor.load_slide_extract_output(mp4_path):
                slide_extractor.extract_slides_from_file(mp4_path)
    return slide_extractor


# Define the streamlit app
def app():
    st.set_page_config(page_title="Recording Slide Extractor", page_icon=":camera_flash:", layout='wide')
    st.write("## Recording Slide Extractor")

    if os.environ.get('HIDE_MENU') == 'true':
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)

    mp4_file_path = ""

    #os.environ["VALID_YOUTUBE_CHANNELS"] = "PyData,Open Data Science,Kaggle,AWS Online Tech Talks,Microsoft Ignite,Microsoft"
    #VALID_YOUTUBE_CHANNELS = os.environ.get("VALID_YOUTUBE_CHANNELS", "").split(",") if os.environ.get("VALID_YOUTUBE_CHANNELS") else []
    
    # Define a file uploader for the user to upload an MP4 file
    st.write("Enter a YouTube link and click the button to download the video.")
    if VALID_YOUTUBE_CHANNELS:
        st.write(f"Supported Channels: {highlight_text(', '.join(VALID_YOUTUBE_CHANNELS))}", unsafe_allow_html=True)
    st.write(f"Max Video Duration: {highlight_text(format_duration(MAX_VIDEO_DURATION))}", unsafe_allow_html=True)
    link = st.text_input("YouTube Link", get_videolink())
    downloaded_mp4 = ""
    st.write(f"If clicking Downwload shows error, try clicking it few times.")
    if st.button("Download"):
        #B
        if is_valid_youtube_link(link) and within_duration_limit(link, duration_limit=MAX_VIDEO_DURATION):
            # either there are no valid channels or the channel is valid
            if (not VALID_YOUTUBE_CHANNELS) or is_valid_channel(link, VALID_YOUTUBE_CHANNELS):
                downloaded_mp4 = download_youtube_video(link, YOUTUBE_VIDEO_DOWNLOAD_PATH)
                st.write("Video Downloaded Successfully!")
                st.session_state['mp4_file_path'] = downloaded_mp4
            else:
                st.warning(f"Invalid Channel. Supported Channels: {', '.join(VALID_YOUTUBE_CHANNELS)}")

    # If the user uploaded an MP4 file, display it using an MP4 player
    if st.session_state.get('mp4_file_path'):
        mp4_file_path = st.session_state.get('mp4_file_path')

        logger.info(f"uploaded file name: {mp4_file_path}")
        video_file = open(st.session_state.get('mp4_file_path'), "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

    if st.session_state.get('mp4_file_path'):
        with_ocr = st.checkbox('Slide Extraction With OCR (takes much longer)', value=False)

    if st.session_state.get('mp4_file_path') and st.button("Run Slide Extraction"):

        slide_extractor = run_slide_extraction(mp4_file_path, with_ocr)
        logger.info(f"ocr and phash and word_ct slides: {len(slide_extractor.new_slide_images)}")
        show_image_files = slide_extractor.new_slide_images
        show_image_files = dict(sorted(show_image_files.items()))
        st.session_state['show_image_files'] = show_image_files
        st.session_state['slide_extractor'] = slide_extractor
    if st.session_state.get('show_image_files') and st.session_state.get('slide_extractor'):
        st.session_state.get('slide_extractor').combined_df.index = st.session_state.get('slide_extractor').combined_df['index']
        metadata_dict = st.session_state.get('slide_extractor').combined_df.to_dict('index')
        show_image_files = st.session_state.get('show_image_files')
        logger.info(f"show images value type: {type(list(show_image_files.values())[0])}")
        files = [
        {
            'name': "Timestamp " + format_duration(image_index),
            'image_index': image_index,
            'duration': format_duration(int(metadata_dict[image_index]['duration']))                
                #'ocr_data': get_paragraph(segmentator.ocr_data_dict[str][0], delimiter=" "),
                #'Timestamp Seconds':str(image_index),
            
        }
        for image_index in show_image_files
        ]
        logger.info(f"post_images download folder: {st.session_state.get('download_folder')}")

        # create a file buffer to store the presentation
        buffer = io.BytesIO()
        prs = create_pptx(show_image_files)
        # save the presentation to the file buffer
        prs.save(buffer)

        st.download_button(
            label="Download Slides to PPT",
            data=buffer.getvalue(),
            file_name=f"{os.path.basename(mp4_file_path).split('.')[0]}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

        # if st.session_state.get('download_folder'):
        #     user_input_folder = st.text_input('Selected folder:', st.session_state.get('download_folder') + "   (DO NOT EDIT)")

        # st.button("Export Images", on_click=export_images)
        display_images(files, st.session_state.get('show_image_files'))
        

app()

