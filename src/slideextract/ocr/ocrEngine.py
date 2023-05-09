from PIL import Image

class OcrEngine(object):
    def __init__(self):
        pass

    def ocr(self, iamge: Image.Image):
        raise NotImplementedError("This method is not implemented yet")
    
    def process_result(self, ocr_result):
        raise NotImplementedError("This method is not implemented yet")
