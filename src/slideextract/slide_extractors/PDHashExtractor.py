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
import logging

from slideextract.slide_extractors.baseSlideExtractor import BaseSlideExtractor
from slideextract.config import PHASH_THRESHOLD, PHASH_PERCENTILE, DHASH_THRESHOLD, DHASH_PERCENTILE, \
    DURATION_THRESHOLD, WORD_CT_THRESHOLD, \
    SLIDE_EXTRACT_OUTPUT_BUCKET_NAME, ocr_new_slide_clf, DATA_FILE_PATH
from slideextract.slide_extractors.image_to_ppt import *


logger = logging.getLogger('pdhashOCRFunnel')

@dataclass
class PDHashExtractor(BaseSlideExtractor):
    duration_threshold: int = DURATION_THRESHOLD
    word_ct_threshold: int = WORD_CT_THRESHOLD
    phash_threshold: int = PHASH_THRESHOLD
    phash_percentile: int = PHASH_PERCENTILE
    dhash_threshold: int = DHASH_THRESHOLD
    frames_data: dict = field(default_factory=dict)
    hash_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    combined_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    new_slide_images: dict = field(default_factory=dict)
    name: str = "pdhash"
    

    def extract_slides_from_frames(self, frames_data: dict):
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
        hash_new_slide_df['is_new_slide'] = True
        hash_new_slide_indices = hash_new_slide_df ['index'].tolist()
        logger.info(f"{len(hash_new_slide_indices)} new slides defined by phash")
        self.hash_df = slide_hash_df

        # export hash_df to csv
        slide_hash_df.to_csv(os.path.join(self.data_file_path,'slide_hash_df.csv'), index=False)

        #self.hash_df['is_new_slide'] = self.hash_df['hash_is_new_slide']

        combined_df = hash_new_slide_df.copy()
        combined_df['duration'] = combined_df['index'].diff(-1).fillna(0) * -1
        combined_df.loc[0,'duration'] = combined_df['index'].min()
        combined_df['duration'] = combined_df['duration'].apply(lambda x : x if x != 0 else 0)
        combined_df['duration_str'] = combined_df['duration'].apply(lambda duration: str(duration//60) + 'min' + str(duration%60)+"s")
        if combined_df.shape[0] > combined_df.dropna().shape[0]:
            logger.warning(f"dropping {combined_df.shape[0] - combined_df.dropna().shape[0]} rows with NaN values")
        combined_df = combined_df.dropna()

        # export combined_df to csv
        self.combined_df = combined_df
        combined_df.index = combined_df['index']
        combined_df.to_csv(os.path.join(self.data_file_path,'combined_df.csv'), index=False)

        new_slide_indices = self.combined_df.query("is_new_slide")['index'].tolist()
        logger.info(f"{len(new_slide_indices)} new slides defined by phash and dhash")
        new_slide_images = dict(filter(lambda record: record[0] in new_slide_indices, frames_data.items()))
        self.new_slide_images = new_slide_images

        # export new_slide_images to files
        for index, image in new_slide_images.items():
            
            new_slide_images_path = os.path.join(self.data_file_path, "new_slide_images")
            if not os.path.exists(new_slide_images_path):
                os.makedirs(new_slide_images_path)
            logger.debug(f"saving new slide image {index} to {os.path.join(new_slide_images_path, f'{index}.jpg')}")
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
            print("data_file_path exists")
            if not os.path.exists(os.path.join(data_file_path, "slide_hash_df.csv")):
                return False
            print("slide_hash_df.csv exists")
            if not os.path.exists(os.path.join(data_file_path, "config.json")):
                return False
            if not os.path.exists(os.path.join(data_file_path, "combined_df.csv")):
                return False
            print("new_slide_images exists")
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

            # load combined_df
            combined_df = pd.read_csv(os.path.join(self.data_file_path,'combined_df.csv'))
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


