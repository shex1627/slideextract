import argparse
import pathlib
import signal
import subprocess
from typing import Dict

from PIL import Image


class FFMPEGException(Exception):
    pass


def sample_video(file_path: str, 
         output_path: str,step_time: int=1, threads: int=0) -> Dict[str, Image.Image]:
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
    worker = subprocess.Popen(list(map(str, args)))

    # stop workers if current process is stopped via (Ctrl+C)
    signal.signal(signal.SIGINT, lambda *_: worker.terminate())

    worker.wait()
    if worker.returncode != 0:
        raise FFMPEGException()

    res = dict()
    for img_path in output.glob('*.jpg'):
        index = int(img_path.stem)
        res[index] = Image.open(img_path)
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