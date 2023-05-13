from PIL import Image
from paddleocr import PaddleOCR
from typing import List, Dict
from functools import partial
import logging
from slideextract.ocr.ocrEngine import OcrEngine
from slideextract.processing.util import image_to_array_gray

logger = logging.getLogger('paddleOcr')

class ppOcr(OcrEngine):
    def __init__(self):
        self.ocr_engine = PaddleOCR()
        return

    def ocr(self, image: Image.Image, to_paragraph: bool=True):
        img_array = image_to_array_gray(image)
        ocr_result = self.ocr_engine.ocr(img_array)
        if to_paragraph:
            return self.to_paragraph(ocr_result)
        return ocr_result
    
    def ocr_batch(self, image_list: list, to_paragraph: bool=False):
        """
        ocr a list of images
        """
        if isinstance(image_list, dict):
            ocr_results = {}
            for key, value in image_list.items():
                result = self.ocr(value, to_paragraph=to_paragraph)
                ocr_results[key] = result
        else:
            func = partial(self.ocr, to_paragraph=to_paragraph)
            ocr_results = list(map(func, image_list))
        
        return ocr_results
        
    def to_paragraph(self, ocr_result, threshold: float = 0.9, delimiter: str = " \n") -> str:
        """
        Processes OCR output into a paragraph of text.
        
        Args:
            ocr_output: OCR output in the format returned by PaddleOCR.
            delimiter: String to use as a delimiter between lines of text. Default is " nextline".
            threshold: Confidence threshold for including a line of text in the output. Default is 0.9.
        
        Returns:
            A paragraph of text containing the OCR output.
            
        Raises:
            ValueError: If the OCR output is not in the expected format.
        """
        try:
            lines = []
            for box, text_conf in ocr_result:
                text, conf = text_conf
                if conf >= threshold:
                    lines.append(text)
            return delimiter.join(lines)
        except Exception as e:
            logger.error(f"OCR output is not in the expected format: {e}")
            return ""