import argparse
import subprocess
import signal
import pathlib
import logging
from typing import Dict
from PIL import Image

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FFMPEGException(Exception):
    pass

class FFMPEGNotFound(FFMPEGException):
    pass

def sample_video(file_path: str, 
         output_path: str,step_time: int=1, threads: int=0,
         sort: bool=True) -> Dict[str, Image.Image]:
    output = pathlib.Path(output_path)
    output.mkdir(exist_ok=True)

    args = [
        "ffmpeg",
        "-threads",
        str(threads),
        "-i",  # input video file
        file_path,
        "-filter_threads",
        str(threads),
        "-threads",
        str(threads),
        "-vf",  # video filter
        # see:  http://ffmpeg.org/ffmpeg-filters.html#Examples-160
        fr"select='isnan(prev_selected_t)+gte(t-prev_selected_t\,{step_time})'",
        "-vsync",
        "vfr",  # variable frame rate
        "-q:v",  # quality
        "2",
        str(output / "%d.jpg")  # output file pattern
    ]
    #worker = subprocess.Popen(list(map(str, args)))
    worker = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # stop workers if current process is stopped via (Ctrl+C)
    stdout, stderr = worker.communicate()
    if worker.returncode != 0:
        logger.error("Error: %s", stderr.decode())  # Log the error message
        raise FFMPEGException()
    else:
        logger.info("Output: %s", stdout.decode())

    res = dict()
    for img_path in output.glob('*.jpg'):
        index = int(img_path.stem)
        res[index] = Image.open(img_path)
    if sort:
        res = {key: res[key] for key in sorted(res.keys())}
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to file to process')
    parser.add_argument('output_path',
                        type=str,
                        help='Path to directory where to store frames.')
    parser.add_argument('step_time',
                        type=int,
                        help='Time between frames to sample (in seconds).')
    parser.add_argument('threads',
                        type=int,
                        help='Number of threads to use for ffmpeg.')
    args = parser.parse_args()
    sample_video(args.file_path, args.step_time, args.threads, args.output_path)