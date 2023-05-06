from collections import OrderedDict
import spacy
from typing import Callable, List
import numpy as np
import pandas as pd
import logging 


nlp = spacy.load('en_core_web_sm')
logger =logging.getLogger("ocr_process")


class docDiffBuilder:
    def __init__(self):
        self.metrics = dict()
        return 

    def add_metric(self, name: str, dist_func: Callable[[str, str], float]):
        """add a metrics to check, no duplicate name allow"""
        assert name not in self.metrics
        self.metrics[name] = dist_func
        return self
    
    def compare(self, doc1:str, doc2: str):
        return {metric_name: dist_func(doc1, doc2) for metric_name, dist_func in self.metrics.items()}

    
    def reset_metric(self):
        self.metric = dict()
        return 0

def make_token_dissim_func(tokenizer: Callable[[str], List[str]], dist_func: Callable[[List[str], List[str]], float]):
    def calculate_dist(doc1:str, doc2:str):
        doc1_tokens = tokenizer(doc1)
        doc2_tokens = tokenizer(doc2)
        return dist_func(doc1_tokens, doc2_tokens)
    return calculate_dist

def jaccard_distance(A, B):
    #Find symmetric difference of two sets
    A, B = set(A), set(B)
    nominator = A.symmetric_difference(B)

    #Find union of two sets
    denominator = A.union(B)

    #Take the ratio of sizes
    if len(denominator) == 0:
        return 0

    distance = len(nominator)/len(denominator)
    
    return distance

def jaccard(list1, list2, dissimilarity):
    """computer jaccard dissimilarity btw two list of tokens"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union == 0:
        return 0
    similarity =  float(intersection) / union
    if dissimilarity:
        return 1 - similarity
    else:
        return similarity


def jaccard_str(doc1: str, doc2:str, tokenizer=lambda doc: doc.lower().split(" "), dissimilarity: bool=True):
    """computer jaccard dissimilarity btw two list of tokens"""
    doc1_tokens = tokenizer(doc1)
    doc2_tokens = tokenizer(doc2)
    metric = jaccard_distance(doc1_tokens, doc2_tokens)
    return metric


def textract_resp_to_paragraph(response, delimiter=" "):
    """
    convert aws textract response object to paragraph form
    """
    response_block_df = pd.DataFrame(response['Blocks'])
    response_block_df['BlockType'].value_counts()
    text_blocks_df = response_block_df[response_block_df['BlockType'].isin(['WORD', 'LINE'])]
    if text_blocks_df.shape[0]:
        return delimiter.join(text_blocks_df['Text'])
    else:
        return ""

word_dissim = make_token_dissim_func(lambda doc: doc.lower().split(" "), 
                      lambda doc1_tokens, doc2_tokens: np.abs(len(doc1_tokens) - len(doc2_tokens))/max(len(doc1_tokens), len(doc2_tokens), 1))
letter_dissim = make_token_dissim_func(lambda doc: list(doc.lower()), 
                      lambda doc1_tokens, doc2_tokens: np.abs(len(doc1_tokens) - len(doc2_tokens))/max(len(doc1_tokens), len(doc2_tokens), 1))

jaccard_letter_dissim = make_token_dissim_func(lambda doc: list(doc.lower()), 
                      lambda doc1_tokens, doc2_tokens: jaccard_distance(doc1_tokens, doc2_tokens))


def get_paragraph(ocr_result: list, threshold: float=0.9, delimiter=" nextline "):
    """parse ocr result into a paragraph.
    User `nextline` as delimiter to add more differences when two paragraph have different lines
    """
    # print("get paragraph")
    # print(ocr_result)
    return delimiter.join(list(map(lambda lst: lst[-1][0],filter(lambda lst: lst[-1][-1] >threshold, ocr_result))))

def get_diff_df(test_ocr_result: dict):
    doc_diff_comparator = docDiffBuilder().\
        add_metric("jaccard_letter", jaccard_letter_dissim).\
        add_metric("jaccard", jaccard_str).\
        add_metric("letter_dis",letter_dissim).\
        add_metric("word_dis", word_dissim)

    test_ocr_paragraph = {int(frame): get_paragraph(ocr_result) for frame, ocr_result in test_ocr_result.items()}
    test_ocr_paragraph = OrderedDict(test_ocr_paragraph)
    img_keys = sorted(test_ocr_paragraph)


    img_keys = sorted(test_ocr_result)
    diff_dict = {}
    for index in range(1, len(img_keys)):
        doc1 = test_ocr_paragraph[img_keys[index]]
        doc2 = test_ocr_paragraph[img_keys[index-1]]
    
        logger.debug(f"index {index}")
        logger.debug(f"doc1 {doc1}")
        logger.debug(f"doc2 {doc2}")

        doc_compare_dict = doc_diff_comparator.compare(doc1, doc2)
        logger.debug(doc_compare_dict)
        frame_token_ct = len(doc1)
        feature_dict = dict(doc_compare_dict)
        feature_dict['frame_token_ct'] = frame_token_ct
        
        diff_dict[img_keys[index]] = feature_dict

    diff_df = pd.DataFrame.from_dict(diff_dict, orient='index').reset_index()
    return diff_df