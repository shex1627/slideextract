import re
from pytube import YouTube
from typing import List
from pptx import Presentation
import os
import streamlit as st
# import tkinter as tk
# from tkinter import filedialog


def format_duration(seconds: int) -> str:
    """
    Convert an integer number of seconds into a duration string in the "HH:MM:SS" format.

    Args:
        seconds: An integer representing the number of seconds to convert.

    Returns:
        A string representing the input number of seconds in the "HH:MM:SS" format.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return duration


def is_valid_youtube_link(url: str) -> bool:
    """
    Check if a string represents a valid YouTube video link.

    Args:
        url: A string representing the YouTube video link to check.

    Returns:
        A boolean value indicating whether the input string is a valid YouTube video link.
    """
    # Check if it is a valid YouTube video link
    regex = (
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/'
        r'(?:watch\?v=)?([a-zA-Z0-9_-]{11})(?:\S+)?'
    )
    match = re.match(regex, url)
    return match is not None

def is_valid_channel(url: str, channels: List[str]) -> bool:
    """
    Check if a YouTube video link belongs to one of the input channels.

    Args:
        url: A string representing the YouTube video link to check.
        channels: A list of strings representing the channel names to check.

    Returns:
        A boolean value indicating whether the YouTube video link belongs to one of the input channels.
    """
    # Check if it is a valid YouTube video link
    yt = YouTube(url)
    channel = yt.author
    print(channel)
    # Extract the channel name from the URL
    if '/' in channel:
        channel = channel.split('/')[-1]
    # Check if the video's channel is in the list of input channels
    if channel in channels:
        return True
    else:
        return False


def within_duration_limit(url: str, duration_limit=3800):
    """
    check if a youtube url exceed duration limit
    duration_limit: in seconds 
    """
    yt = YouTube(url)
    duration_in_seconds = yt.length
    return duration_in_seconds <= duration_limit

def highlight_text(word_to_highlight: str):
    """
    Returns the text with the word highlighted using HTML tags.
    """
    highlighted_text = f"<span style='background-color: #98FB98;'>{word_to_highlight}</span>"
    highlighted_text =  f"<b>{word_to_highlight}</b>"
    return highlighted_text

# Create a function to save the presentation to a file
def save_presentation(presentation, file_path):
    presentation.save(file_path)


import re
import streamlit as st

def get_videolink():
    # Get the query parameter from the URL
    query_params = st.experimental_get_query_params()
    video_id = query_params.get('video_id', [''])[0]

    # Define the regex pattern for matching YouTube video IDs
    YOUTUBE_ID_PATTERN = r'^[a-zA-Z0-9_-]{11}$'

    # Check if video_id matches the YouTube video ID pattern
    if not re.match(YOUTUBE_ID_PATTERN, video_id):
        return ''
    else:
        video_link = f'https://www.youtube.com/watch?v={video_id}'
        return video_link
