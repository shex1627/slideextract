import boto3
from typing import List, Dict
import imagehash
from PIL import Image
import cv2
import logging
import io
import os
import time
import numpy as np
import joblib
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger("slide_extractor_util")

def process_chunk(video_path, chunk, fps):
    """
    Reads a chunk of frames from a video file and returns them as a dictionary of numpy arrays.

    Args:
        video_path: The path to the input video file.
        chunk: A numpy array of frame indices to read.
        fps: The frames per second of the input video.

    Returns:
        A dictionary of frames as numpy arrays. The keys are frame indices divided by the video's frames per second,
        and the values are the corresponding frame images in numpy array format.
    """
    cap = cv2.VideoCapture(video_path)
    frames = {}
    for frame_index in chunk:
        # Set the video frame index and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        # Add the frame to the dictionary
        key = int(frame_index // fps)
        frames[key] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def sample_video_from_file(mp4_file_path: str, interval: int = 1, num_workers: int = 4) -> Dict[float, np.ndarray]:
    """
    use opencv to load the mp4 file, sample images per second interval and return a dictionary 
    where the keys are the time in seconds and the values are the corresponding frames as 
    numpy arrays
    """
    """
    Sample frames from a video and return them as a dictionary of numpy arrays.

    Args:
        video_path: The path to the input video file.
        interval: The interval in seconds between sampled frames. Defaults to 1.
        num_workers: The number of parallel workers to use. Defaults to 4.

    Returns:
        A dictionary of sampled frames as numpy arrays. The keys are frame indices divided by the video's frames per second,
        and the values are the corresponding frame images in numpy array format.
    """

    # Open the video file and get its properties
    cap = cv2.VideoCapture(mp4_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the indices of frames to sample
    frame_indices = np.arange(0, num_frames, int(interval * fps))

    # Divide the frame indices into chunks for parallel processing
    chunk_size = len(frame_indices) // num_workers
    chunks = [frame_indices[i:i+chunk_size] for i in range(0, len(frame_indices), chunk_size)]

    # Use joblib to process the chunks in parallel
    frames_list = joblib.Parallel(n_jobs=num_workers, backend='multiprocessing')(
        joblib.delayed(process_chunk)(mp4_file_path, chunk, fps) for chunk in chunks
    )

    # Aggregate the frames from all chunks into a single dictionary
    frames = {}
    for f in frames_list:
        frames.update(f)

    # Release the video file and return the frames
    cap.release()
    return frames


def list_jpg_files_in_s3_folder(s3_client: boto3.client, s3_bucket: str, s3_folder: str = "") -> List[str]:
    """
    Lists all the JPG files in an S3 folder.

    :param s3_client: A Boto3 S3 client.
    :param s3_bucket: The name of the S3 bucket.
    :param s3_folder: The path of the S3 folder, defaults to "".
    :return: A list of JPG file names.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_folder)
        files = [obj['Key'] for obj in response['Contents'] if obj['Key'].lower().endswith('.jpg')]
        return files
    except ClientError as e:
        print(e)
        return []


def upload_images_to_s3(indices, img_dict: Dict, s3_client, s3_bucket: str, s3_folder=""):
    """
    indices: indices of the img_dict
    img_dict: dictionary key is an integer index, value is an Image object
    
    for each index in indices, upload the image to designated s3 bucket and folder, 
    where the name of the file is index.jpg
    """
    for index in indices:
        image = img_dict[index]
        file_name = str(index) + ".jpg"
        s3_key = s3_folder + "/" + file_name if s3_folder else file_name
        with io.BytesIO() as output:
            image.save(output, format='JPEG')
            output.seek(0)
            s3_client.upload_fileobj(output, s3_bucket, s3_key)

def upload_dataframe_to_s3(s3_client, dataframe: pd.DataFrame, bucket_name: str, folder_path: str, file_name: str) -> None:
    """
    Uploads a Pandas DataFrame to a designated S3 bucket and folder.

    Args:
        dataframe (pd.DataFrame): The Pandas DataFrame to be uploaded.
        bucket_name (str): The name of the S3 bucket to upload the DataFrame to.
        folder_path (str): The path of the folder within the S3 bucket to upload the DataFrame to.
        file_name (str): The name of the file to be saved in S3.

    Returns:
        None
    """
    # Create an S3 client
    #s3_client = boto3.client('s3')

    # Create a buffer to hold the CSV data
    csv_buffer = io.StringIO()

    # Write the DataFrame to the CSV buffer
    dataframe.to_csv(csv_buffer, index=False)

    # Generate the S3 object key using the folder path and file name
    object_key = f"{folder_path}/{file_name}"

    # Upload the CSV data to S3
    s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=bucket_name, Key=object_key)


def run_textract_batch(textract_client, s3_bucket: str, file_list: List):
    chunk_size = 90
    start = time.time()
    output = {}
    
    # Break the file list into chunks with max size of 90
    file_chunks = [file_list[i:i+chunk_size] for i in range(0, len(file_list), chunk_size)]
    logger.info(f"there are {len(file_chunks)} chunks")
    
    # Send async text detection request to Textract for each chunk
    for file_chunk in file_chunks:
        # Create a list to hold the response objects for the async request
        logger.info(f"processing chunk, {time.time()-start:.2f} seconds passed")
        index_to_job_ids = dict()

        # Loop through each file in the file chunk
        for file in file_chunk:
            image_index = int(os.path.splitext(os.path.basename(file))[0])
            # Construct the S3 object key from the file name and the bucket name
            s3_object_key = f'{file}'
            # Call the Textract async text detection API with the S3 object as input
            response = textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': s3_bucket,
                        'Name': s3_object_key
                    }
                }
            )
            job_id = response['JobId']
            index_to_job_ids[image_index] = job_id


        logger.info(f"waiting for job to finish, {time.time()-start:.2f} seconds passed")
        time.sleep(5)
        for image_index, job_id in index_to_job_ids.items():
            response = textract_client.get_document_text_detection(
                JobId=job_id
            )
            while response['JobStatus'] == 'IN_PROGRESS':
                response = textract_client.get_document_text_detection(
                    JobId=job_id
                )
                logger.info(f"waiting for {job_id}")
                time.sleep(1)
            # Extract base filename without suffix
            output[image_index] = response
        
    end = time.time()
    logger.info(f"OCR processing completed in {end-start:.2f} seconds")
    
    return output


def s3_folder_exists(bucket_name: str, folder_name: str) -> bool:
    """
    Checks whether an S3 folder (i.e., prefix) exists in the specified S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_name (str): The name of the folder (i.e., prefix) to check.

    Returns:
        bool: True if the folder exists in the bucket, False otherwise.

    Raises:
        botocore.exceptions.NoCredentialsError: If AWS credentials are not configured properly.
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    prefix = folder_name + '/'
    objects = list(bucket.objects.filter(Prefix=prefix))
    #logger.info(f"objects found {objects}")
    return len(objects) > 0