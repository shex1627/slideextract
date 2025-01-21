import yt_dlp
import os
from pathlib import Path
from typing import Dict, Any
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)
import re
from yt_dlp import YoutubeDL

def download_youtube_video(link, output_path: str = "./youtube_downloads"):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Best quality MP4
        'merge_output_format': 'mp4',
        'quiet': True,  # Reduces console output
        'no_warnings': True,
        'extract_flat': True,  # Just extract video metadata first
    }
    
    # First get the video title
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        title = info['title'].lower().replace(" ", "_")
        filename = re.sub('[^A-Za-z0-9_]+', '', title) + ".mp4"
        full_path = os.path.join(output_path, filename)
        
        # Update options with output path
        ydl_opts.update({
            'outtmpl': full_path,
            'extract_flat': False,  # Now actually download
        })
    
    # Download the video
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    
    return full_path

class YouTubeDownloader:
    def __init__(self):
        #self.download_dir = download_dir #Path("temp_downloads")
        self._setup_options()

    def _setup_options(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'vorbis',
            }],
            'outtmpl': os.path.join(os.getcwd(), "downloaded_song.ogg"),
            'quiet': True,
            'no_warnings': True,
        }

    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """Extract video information without downloading."""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as e:
            logger.error(f"Error extracting video info: {str(e)}")
            raise

    def download_audio(self, url:str, output_filepath: str) -> Path:
        """
        Downloads audio from a URL and converts it to an OGG file with the Vorbis codec.
        
        Parameters:
        url (str): URL of the audio or video to download.
        filepath (str): Path to save the downloaded file.
        
        Returns:
        None
        """
        # outputfile is current directory and name is downloaded_song
        output_file = os.path.join(os.getcwd(), "downloaded_song.ogg")
        try:
            # Build the yt-dlp command
            yt_dlp_command = [
                'yt-dlp', 
                '-x', 
                '--audio-format', 'vorbis', 
                '-o', "downloaded_song", 
                url
            ]
            
            # Execute the yt-dlp command
            subprocess.run(yt_dlp_command, check=True)
            # check if output file exists, if so, copy it to the desired output path
            if os.path.exists(output_file):
                shutil.copy(output_file, output_filepath)
                os.remove(output_file)
                print(f"Download and conversion complete: {url} to {output_filepath}")
                return output_filepath
            else:
                print(f"Download and conversion failed: {url}")
                return None
        except subprocess.CalledProcessError as e:
            print(f"Error during download and conversion: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
        
    def download_video(self, url: str, output_filepath: Path) -> Path:
        """Download video from a URL and save it to the specified output path."""
            # Configure the options
        ydl_opts = {
            'format': 'mp4',  # Specify MP4 format
            'outtmpl': f'{output_filepath}/%(title)s.%(ext)s',  # Output template
        }
        
        try:
            # Create a yt-dlp object with our options
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the video
                ydl.download([url])
            print("Download completed successfully!")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    async def async_download_audio(self, url: str, output_filepath: Path) -> Path:
        """Download audio from a URL and save it to the specified output path."""
        return self.download_audio(url, output_filepath)