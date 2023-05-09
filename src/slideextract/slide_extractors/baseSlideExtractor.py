import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import cv2
import numpy as np
import time
import pandas as pd
import imagehash
import multiprocessing as mp
import logging
import io
from typing import List, Dict

from slideextract.config import ocr_feature_names, ocr_new_slide_clf
from slideextract.comparator.docDiffBuilder import docDiffBuilder, doc_diff_comparator
from slideextract.processing.ffmpeg_sample_video import sample_video


logger = logging.getLogger('baseSlide')

class BaseSlideExtractor:
    """
    base class for slide extractor,
    all slide extractor should inherit from this class
    have general methods to extract images from video
    compare slides based on imagebased feature and ocr feature
    """
    def __init__(self, *args, **kwargs) -> None:
        pass 

    def extract_slides(self, mp4_file_path: str):
        NotImplementedError

    def extract_slides_from_file(self, mp4_file_path: str, threads: int = 0):
        frames_data = sample_video(mp4_file_path, threads=threads)
        return self.extract_slides_from_frames(frames_data)

    def extract_slides_from_frames(self, frames_data: dict):
        NotImplementedError

    def _generate_ocr_doc_feature(self, ocr_paragraph1: str, ocr_paragraph2: str, doc_diff_comparator: docDiffBuilder=doc_diff_comparator):
        """
        generate feature feature based on ocr results
        """
        doc1 = ocr_paragraph1
        doc2 = ocr_paragraph2
        doc_compare_dict = doc_diff_comparator.compare(doc1, doc2)
        doc_compare_dict['frame_token_ct'] = max([len(doc1), len(doc2)])
        # need to test if dataframe results results
        feature_df = pd.DataFrame([doc_compare_dict])
        feature_df = feature_df.rename(columns={'letter_dis':'letter_dissim'})
        return feature_df[ocr_feature_names] 
    
    def _compare_ocr_results(self, ocr_slide_indices: List, ocr_paragraphs: Dict, clf_model) -> pd.DataFrame:
        ocr_slide_record = []
        for index in ocr_slide_indices:
            if index > 0:
                feature_df = \
                    self._generate_ocr_doc_feature(
                        ocr_paragraph1=ocr_paragraphs[index], ocr_paragraph2=ocr_paragraphs[index-1])
                ocr_is_new_slide = ocr_new_slide_clf.predict(feature_df)[0]
                ocr_slide_record.append((index, ocr_is_new_slide))

        ocr_new_slide_df = pd.DataFrame(ocr_slide_record)
        ocr_new_slide_df.columns = ['index', 'ocr_is_new_slide']
        return ocr_new_slide_df
    
    def _classify_if_ocr_same(self, feature_df: pd.DataFrame, clf_model) -> bool:
        """
        classify if ocr results are the same
        """
        return clf_model.predict(feature_df)[0]
    
    @classmethod
    def compare_frames(frames, comparators):
        """
        Use the output of 1, and a list of python callable[(image1, image2), float],
        return the dataframe with the following columns:
        index: index of the frame
        phash: percetual hash of the image with previous frame (create phash comparater)
        dhash: dhash diff of the image with previous frame (create dhash comparater)
        """
        data = []
        prev_frame = None
        for index, frame in frames.items():
            row = {"index": index}
            if prev_frame is not None:
                for comparator, name in comparators:
                    row[name] = comparator(prev_frame, frame)
            data.append(row)
            prev_frame = frame
        return pd.DataFrame(data)
