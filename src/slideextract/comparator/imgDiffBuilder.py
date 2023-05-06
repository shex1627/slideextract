import os
import cv2
import imagehash
import pandas as pd
from typing import Callable, Tuple
from PIL import Image

import logging

logger = logging.getLogger("imgDiffBuilder")

comparators = [
        (lambda x, y: imagehash.phash(x) - imagehash.phash(y), "phash"),
        (lambda x, y: imagehash.dhash(x) - imagehash.dhash(y), "dhash"),
        (lambda x, y: cv2.medianBlur(cv2.absdiff(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), 
                                                 cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)), 1).mean(), "mse")
]

class imgDiffBuilder:
    def __init__(self):
        self.metrics = dict()
        return 

    def add_metric(self, name: str, dist_func: Callable[[any, any], float]):
        """add a metrics to check, no duplicate name allow"""
        assert name not in self.metrics
        self.metrics[name] = dist_func
        return self
    
    def compare(self, obj1:str, obj2: str):
        return {metric_name: dist_func(obj1, obj2) for metric_name, dist_func in self.metrics.items()}

    
    def reset_metric(self):
        self.metric = dict()
        return 0