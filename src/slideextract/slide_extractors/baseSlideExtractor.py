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

from slide_segmentation.ocr.ocr_process import docDiffBuilder, textract_resp_to_paragraph
from slide_segmentation.ocr.config import doc_diff_comparator, new_slide_clf, feature_names


logger = logging.getLogger('baseSlide')

def _compute_phash(idx, prev_img: np.array, curr_img: np.array):
    phash = imagehash.phash(curr_img)
    prev_phash = imagehash.phash(prev_img)
    return (idx, phash - prev_phash)

class BaseSlideExtractor:
    def __init__(self, *args, **kwargs) -> None:
        pass 

    def _compute_phash_difference(self, img_dict: Dict[int, np.ndarray], n_jobs = 1) -> pd.DataFrame:
        if n_jobs > 1:
            with mp.Pool(n_jobs) as pool:
                print("start computing phash Pool")
                results = pool.starmap(_compute_phash, [(i, img_dict[i-1], img_dict[i]) for i in range(1, len(img_dict))])
        else:
            results = [_compute_phash(i, img_dict[i-1], img_dict[i]) for i in range(1, len(img_dict))]

        df = pd.DataFrame(results, columns=["index", "phash_diff"])
        df.set_index("index", inplace=True)
        return df

    def extract_slides(self, mp4_file_path: str):
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
        return feature_df[feature_names] 
    
    def _compare_ocr_results(self, ocr_slide_indices: List, ocr_paragraphs: Dict, clf_model) -> pd.DataFrame:
        ocr_slide_record = []
        for index in ocr_slide_indices:
            if index > 0:
                feature_df = \
                    self._generate_ocr_doc_feature(
                        ocr_paragraph1=ocr_paragraphs[index], ocr_paragraph2=ocr_paragraphs[index-1])
                ocr_is_new_slide = new_slide_clf.predict(feature_df)[0]
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
