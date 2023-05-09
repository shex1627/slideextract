from typing import Dict
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import joblib
import cv2
import numpy as np
import time
import pandas as pd
import imagehash
import json
from dataclasses import dataclass, field


from slideextract.processing.ffmpeg_sample_video import sample_video
import logging

from slideextract.slide_extractors.baseSlideExtractor import BaseSlideExtractor
from slideextract.config import PHASH_THRESHOLD, PHASH_PERCENTILE, DHASH_THRESHOLD, DHASH_PERCENTILE, \
    DURATION_THRESHOLD, WORD_CT_THRESHOLD, \
    SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, new_slide_clf
from slideextract.slide_extractors.image_to_ppt import *


logger = logging.getLogger('pdhashOCRFunnel')

@dataclass
class PDhashOCRFunnel(BaseSlideExtractor):
    duration_threshold: int = DURATION_THRESHOLD
    word_ct_threshold: int = WORD_CT_THRESHOLD
    phash_threshold: int = PHASH_THRESHOLD
    phash_percentile: int = PHASH_PERCENTILE
    dhash_threshold: int = DHASH_THRESHOLD
    frames_data: dict = field(default_factory=dict)
    phash_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    combined_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    new_slide_images: dict = field(default_factory=dict)
    ocr_data_dict: dict = field(default_factory=dict)

    def extract_slides_from_frames(self, frames_data: dict):
        pass

    def extract_slides(self, mp4_file_path: str):
        logger.info("start sampling images from file")
        frames_data = sample_video_from_file(mp4_file_path, num_workers=1)
        self.frames_data = frames_data
        logger.info("done sampling images from file")

        logger.info("start computing phash")
        slide_phash_df = self._compute_phash_difference(frames_data, n_jobs=1)
        slide_phash_df = slide_phash_df.reset_index()
        slide_phash_df.columns = ['index', 'phash_diff']
        logger.info("end computing phash")

        percentile_threshold = slide_phash_df['phash_diff'].quantile(PHASH_PERCENTILE)
        threshold = min(percentile_threshold , PHASH_THRESHOLD)
        logger.info(f"final threshold: {threshold}, percentile threshold: {percentile_threshold}, predefine threshold: {PHASH_THRESHOLD}")
        slide_phash_df['phash_is_new_slide'] = slide_phash_df['phash_diff'] > threshold
        phash_new_slide_df = slide_phash_df.query("phash_is_new_slide")
        phash_new_slide_indices = phash_new_slide_df ['index'].tolist()
        logger.info(f"{len(phash_new_slide_indices)} new slides defined by phash")

        self.phash_df = slide_phash_df

        ## aws stuff
        session = boto3.Session()
        s3_client = session.client("s3")
        s3_resource = boto3.resource('s3')
        s3_bucket = s3_resource.Bucket(name=SLIDE_EXTRACT_OUTPUT_BUCKET_NAME)
        textract_client = session.client('textract')
        folder_name = os.path.basename(mp4_file_path).split(".")[0]
        #image_folder_name = os.path.join(folder_name, "phash_images")
        image_folder_name = folder_name+"/phash_images"

        upload_to_s3_indices = set(phash_new_slide_indices + [max(0, index - 1) for index in phash_new_slide_indices])
        logger.info(f"uploading {len(upload_to_s3_indices)} images to s3 for ocr")
        upload_images_to_s3(upload_to_s3_indices, frames_data, s3_client, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME , image_folder_name)
        logger.info(f"finished uploading images to s3")

        list_s3_for_ocr = list_jpg_files_in_s3_folder(s3_client, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, image_folder_name)
        ocr_results = run_textract_batch(textract_client, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, list_s3_for_ocr)
        logger.info(f"found {len(ocr_results)} ocr records")
        # this may duplicate with ocr_paragraph 
        ocr_paragraphs = {image_index: textract_resp_to_paragraph(ocr_result) for image_index, ocr_result in ocr_results.items()}
        ocr_paragraphs_token_ct = \
            {image_index: len(textract_resp_to_paragraph(ocr_result).split(" ")) for image_index, ocr_result in ocr_results.items()}
        
        # combining the results
        ocr_new_slide_df = self._compare_ocr_results(phash_new_slide_indices, ocr_paragraphs, new_slide_clf)
        phash_new_slide_df['img_word_ct'] = phash_new_slide_df.copy()['index'].apply(lambda index: ocr_paragraphs_token_ct[index])
        phash_ocr_combined_df = phash_new_slide_df.merge(ocr_new_slide_df, on='index')
        combined_df = phash_ocr_combined_df.query("ocr_is_new_slide").copy()
        combined_df['duration'] = combined_df.copy()['index'].diff(1).fillna(combined_df['index'].min())
        combined_df['duration_str'] = combined_df['duration'].apply(lambda duration: str(duration//60) + 'min' + str(duration%60)+"s")
        combined_df['enough_words_in_slide'] = combined_df['img_word_ct'] >= self.word_ct_threshold
        combined_df['enough_duration_in_slide'] = combined_df['duration'] >= self.duration_threshold

        combined_df['is_new_slide'] = combined_df['enough_words_in_slide'] & combined_df['enough_duration_in_slide']
        combined_df.index = combined_df['index']
        logger.info(f"phash new slides: {phash_ocr_combined_df.shape}, ocr_new_slide: {combined_df.shape}, filtered_new_slide: {combined_df.query('is_new_slide').shape}")
        self.combined_df = combined_df
        upload_dataframe_to_s3(s3_client, slide_phash_df, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME,
                      folder_name, "phash_df.csv")
        upload_dataframe_to_s3(s3_client, phash_ocr_combined_df, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME,
                      folder_name, "phash_ocr_combined_df.csv")
        
        upload_dataframe_to_s3(s3_client, combined_df, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME,
                      folder_name, "combined_df.csv")
        
        data_string = json.dumps(ocr_paragraphs, indent=4, default=str)

        s3_bucket.put_object(
            Key=f'{folder_name}/ocr_paragraphs.json',
            Body=data_string
        )
        self.ocr_data_dict = ocr_paragraphs

        new_slide_indices = list(combined_df.query("is_new_slide").index)
        new_slide_images = dict(filter(lambda record: record[0] in new_slide_indices, frames_data.items()))
        self.new_slide_images = new_slide_images
        pptx_obj = create_pptx(new_slide_images)
        upload_pptx_to_s3(pptx_obj, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, folder_name, 
                  title=os.path.basename(mp4_file_path).split(".")[0])
        return pptx_obj
    

    def check_if_slides_exist(self, mp4_file_path: str) -> bool:
        folder_name = os.path.basename(mp4_file_path).split(".")[0]
        logger.info(f"checking if {SLIDE_EXTRACT_OUTPUT_BUCKET_NAME}/{folder_name} exists")
        return s3_folder_exists(SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, folder_name)
    
    def load_slide_extract_output(self, mp4_file_path: str) -> None:
        """
        give an mp4 file path, load the results of slide extraction if that exists already.
        return True if dataloaded
        return False if data not loaded
        """
        if self.check_if_slides_exist(mp4_file_path):
            logger.info("start loading pre-computed slide results")
            # to do-refactor those
            folder_name = os.path.basename(mp4_file_path).split(".")[0]
            image_folder_name = folder_name+"/phash_images"

            session = boto3.Session()
            s3_client = session.client("s3")

            # get combined_df info
            response = s3_client.get_object(Bucket=SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, Key=f"{folder_name}/combined_df.csv")
            combined_df = pd.read_csv(response['Body'])

            # response = s3_client.get_object(Bucket=SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, Key=f"{folder_name}/phash_df.csv")
            # slide_phash_df = pd.read_csv(response['Body'])

            new_slide_indices = combined_df.query("is_new_slide")['index'].tolist()

            list_s3_for_ocr = list_jpg_files_in_s3_folder(s3_client, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, image_folder_name)
            logger.info(f"downloading {len(list_s3_for_ocr)} images from s3")
            new_slide_images = {}
            # Download each object using the key and store the contents in a dictionary
            for key in list_s3_for_ocr:
                try:
                    basename = os.path.basename(key)
                    image_index = int(basename.split(".")[0])
                    if key.endswith('.jpg') and image_index in new_slide_indices:
                        response = s3_client.get_object(Bucket=SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, Key=key)
                        content = response['Body'].read()
                        #new_slide_images[image_index] = content
                        new_slide_images[image_index] = Image.open(io.BytesIO(content))
                except Exception as e:
                    logger.error(f"error processing key {key}")
                    logger.error(e, exc_info=True)
            new_slide_images = dict(sorted(new_slide_images.items()))
            self.new_slide_images = new_slide_images
            combined_df['index'] = combined_df['index'].apply(int)
            combined_df.index = combined_df['index'] 
            self.combined_df = combined_df
            #self.phash_df = slide_phash_df
            return True
        else:
            logger.info("found no pre-computed slide results")
            return False

