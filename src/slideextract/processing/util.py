from PIL import Image
import io
import numpy as np
from typing import List, Dict

def image_to_array(image: Image.Image):
    """
    convert a PIL image to a numpy array
    """
    return np.array(image)

def array_to_image(array: np.ndarray):
    """
    convert a numpy array to a PIL image
    """
    return Image.fromarray(array)

def image_to_array_gray(image: Image.Image):
    """
    convert a PIL image to grayscale then a numpy array
    """
    return np.array(image.convert('L'))


def image_to_bytes(image: Image.Image):
    """convert a PIL image to bytes for aws textract service"""
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image_bytes = output.getvalue()
    return image_bytes



def chunk_images(images: List[str], chunk_size: int) -> List[List[str]]:
    """
    split a list of images into chunks of size chunk_size
    """
    image_chunks = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]
    return image_chunks