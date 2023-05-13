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


from slideextract.comparator.imgDiffBuilder import pd_hash_img_diffbuilder as img_diff_builder
from slideextract.ocr.textractOcr import textractOcr
from slideextract.ocr.ocrEngine import OcrEngine
import logging

from slideextract.slide_extractors.baseSlideExtractor import BaseSlideExtractor
from slideextract.config import PHASH_THRESHOLD, PHASH_PERCENTILE, DHASH_THRESHOLD, DHASH_PERCENTILE, \
    DURATION_THRESHOLD, WORD_CT_THRESHOLD, \
    SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, ocr_new_slide_clf, DATA_FILE_PATH
from slideextract.slide_extractors.image_to_ppt import *


logger = logging.getLogger('pdhashOCRFunnel')

@dataclass
class PDHashOCRFunnel(BaseSlideExtractor):
    ocr_engine: OcrEngine
    duration_threshold: int = DURATION_THRESHOLD
    word_ct_threshold: int = WORD_CT_THRESHOLD
    phash_threshold: int = PHASH_THRESHOLD
    phash_percentile: int = PHASH_PERCENTILE
    dhash_threshold: int = DHASH_THRESHOLD
    frames_data: dict = field(default_factory=dict)
    hash_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    combined_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    new_slide_images: dict = field(default_factory=dict)
    ocr_data_dict: dict = field(default_factory=dict)
    name: str = "pdhashOCRFunnel"


    def extract_slides_from_frames(self, frames_data: dict):
        """
        to-do, setup self.data_file_path with a random value when self.data_file_path is None
        """
        if self.data_file_path is None:
            self.data_file_path = os.path.join(DATA_FILE_PATH, str(int(time.time())), self.name)
            os.makedirs(self.data_file_path, exist_ok=True)
            logger.info(f"self.data_file_path is None, set to {self.data_file_path}")

        logger.info("start computing phash")
        slide_hash_df = img_diff_builder.compare_images(frames_data)
        logger.info("end computing phash")

        percentile_threshold = slide_hash_df['phash'].quantile(PHASH_PERCENTILE)
        final_phash_threshold = min(percentile_threshold , PHASH_THRESHOLD)
        logger.info(f"final threshold: {final_phash_threshold}, percentile threshold: {percentile_threshold}, predefine threshold: {PHASH_THRESHOLD}")

        # export config to json
        config_dict = {
            "duration_threshold": self.duration_threshold,
            "word_ct_threshold": self.word_ct_threshold,
            "phash_threshold": self.phash_threshold,
            "phash_percentile": self.phash_percentile,
            "final_phash_threshold": final_phash_threshold,
            "dhash_threshold": self.dhash_threshold,
        }
        with open(os.path.join(self.data_file_path,'config.json'), 'w') as fp:
            json.dump(config_dict, fp, indent=4)

        slide_hash_df['hash_is_new_slide'] = (slide_hash_df['phash'] > final_phash_threshold) | (slide_hash_df['dhash'] > DHASH_THRESHOLD)
        hash_new_slide_df = slide_hash_df.query("hash_is_new_slide")
        hash_new_slide_indices = hash_new_slide_df ['index'].tolist()
        logger.info(f"{len(hash_new_slide_indices)} new slides defined by phash")
        self.hash_df = slide_hash_df

        # export hash_df to csv
        slide_hash_df.to_csv(os.path.join(self.data_file_path,'slide_hash_df.csv'), index=False)

        upload_to_ocr_indices = set(hash_new_slide_indices + [max(0, index - 1) for index in hash_new_slide_indices])
        hash_images = {index: frames_data[index] for index in upload_to_ocr_indices}
        
        ocr_results = self.ocr_engine.ocr_batch(hash_images)
        ocr_paragraphs = {int(index): self.ocr_engine.to_paragraph(ocr_result) for index, ocr_result in ocr_results.items()}
        ocr_paragraphs_token_ct = {index: len(result.split(" ")) for index, result in ocr_paragraphs.items()}
        self.ocr_paragraphs = ocr_paragraphs

        # export paragraphs to json
        with open(os.path.join(self.data_file_path,'ocr_paragraphs.json'), 'w') as fp:
            json.dump(ocr_paragraphs, fp, indent=4)

        # combining the results
        if len(hash_new_slide_indices) == 0:
            logger.info("no new slides detected")
            return
        ocr_new_slide_df = self._compare_ocr_results(hash_new_slide_indices, ocr_paragraphs, ocr_new_slide_clf)
        hash_new_slide_df['img_word_ct'] = hash_new_slide_df.copy()['index'].apply(lambda index: ocr_paragraphs_token_ct[index])
        hash_ocr_combined_df = hash_new_slide_df.merge(ocr_new_slide_df, on='index')

        # export hash_ocr_combined_df to csv
        hash_ocr_combined_df.to_csv(os.path.join(self.data_file_path,'hash_ocr_combined_df.csv'), index=False)

        combined_df = hash_ocr_combined_df.query("ocr_is_new_slide").copy()
        combined_df['duration'] = combined_df['index'].diff(-1).fillna(combined_df['index'].min()) * -1
        combined_df.loc[0,'duration'] = combined_df['index'].min()
        combined_df['duration_str'] = combined_df['duration'].apply(lambda duration: str(duration//60) + 'min' + str(duration%60)+"s")
        combined_df['enough_words_in_slide'] = combined_df['img_word_ct'] >= self.word_ct_threshold
        combined_df['enough_duration_in_slide'] = combined_df['duration'] >= self.duration_threshold

        combined_df['is_new_slide'] = combined_df['enough_words_in_slide'] & combined_df['enough_duration_in_slide']
        combined_df.index = combined_df['index']
        logger.info(f"hash new slides: {hash_ocr_combined_df.shape}, ocr_new_slide: {combined_df.shape}, filtered_new_slide: {combined_df.query('is_new_slide').shape}")
        self.combined_df = combined_df
        self.ocr_data_dict = ocr_paragraphs

        # export combined_df to csv
        combined_df.to_csv(os.path.join(self.data_file_path,'combined_df.csv'), index=False)

        new_slide_indices = list(combined_df.query("is_new_slide").index)
        new_slide_images = dict(filter(lambda record: record[0] in new_slide_indices, frames_data.items()))
        logger.info(f"new slides: {len(new_slide_images)}")
        self.new_slide_images = new_slide_images

        # export new_slide_images to files
        for index, image in new_slide_images.items():
            new_slide_images_path = os.path.join(self.data_file_path, "new_slide_images")
            if not os.path.exists(new_slide_images_path):
                os.makedirs(new_slide_images_path)
            image.save(os.path.join(new_slide_images_path, f"{index}.jpg"))

        pptx_obj = create_pptx(new_slide_images)
        # upload_pptx_to_s3(pptx_obj, SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, folder_name, 
        #           title=os.path.basename(mp4_file_path).split(".")[0])

        return pptx_obj
    
    def check_if_slides_exist(self, mp4_file_path: str) -> bool:
        """
        give an mp4 file path, check if the results of slide extraction exists already.
        return True if exists
        return False if not exists
        """
        filename = os.path.basename(mp4_file_path).split(".")[0]
        data_file_path = os.path.join(DATA_FILE_PATH, filename, self.name)
        if os.path.exists(data_file_path):
            if not os.path.exists(os.path.join(data_file_path, "slide_hash_df.csv")):
                return False
            if not os.path.exists(os.path.join(data_file_path, "ocr_paragraphs.json")):
                return False
            if not os.path.exists(os.path.join(data_file_path, "hash_ocr_combined_df.csv")):
                return False
            if not os.path.exists(os.path.join(data_file_path, "combined_df.csv")):
                return False
            if not os.path.exists(os.path.join(data_file_path, "new_slide_images")):
                return False
            return True
        else:
            return False
                          
    def load_slide_extract_output(self, mp4_file_path: str) -> None:
        """
        give an mp4 file path, load the results of slide extraction if that exists already.
        return True if dataloaded
        return False if data not loaded
        """
        if self.check_if_slides_exist(mp4_file_path):
            logger.info("start loading pre-computed slide results")
            # to do-refactor those
            filename = os.path.basename(mp4_file_path).split(".")[0]
            self.data_file_path = os.path.join(DATA_FILE_PATH, filename, self.name)

            # load slide_hash_df
            slide_hash_df = pd.read_csv(os.path.join(self.data_file_path,'slide_hash_df.csv'))
            slide_hash_df.index = slide_hash_df['index']
            self.slide_hash_df = slide_hash_df

            # load hash_new_slide_df
            hash_new_slide_df = pd.read_csv(os.path.join(self.data_file_path,'hash_ocr_combined_df.csv'))
            hash_new_slide_df.index = hash_new_slide_df['index']
            self.hash_new_slide_df = hash_new_slide_df

            # load ocr_paragraphs
            with open(os.path.join(self.data_file_path,'ocr_paragraphs.json'), 'r') as fp:
                ocr_paragraphs = json.load(fp)
            self.ocr_data_dict = ocr_paragraphs


            # load combined_df
            combined_df = pd.read_csv(os.path.join(self.data_file_path,'combined_df.csv'))
            combined_df.index = combined_df['index']
            self.combined_df = combined_df

            # load new_slide_images
            new_slide_images = {}
            new_slide_images_path = os.path.join(self.data_file_path, "new_slide_images")
            for image_file in os.listdir(new_slide_images_path):
                image = Image.open(os.path.join(new_slide_images_path, image_file))
                index = int(image_file.split(".")[0])
                new_slide_images[index] = image
            new_slide_images = dict(sorted(new_slide_images.items()))
            self.new_slide_images = new_slide_images
            return True
        else:
            logger.info("found no pre-computed slide results")
            return False


