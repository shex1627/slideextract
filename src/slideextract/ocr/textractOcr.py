from PIL import Image
from typing import List, Union, Dict 
import logging
import boto3
import io
import os 
import time

from slideextract.ocr.ocrEngine import OcrEngine
from slideextract.processing.util import image_to_bytes, chunk_images

logger = logging.getLogger('textractOcr')

class textractOcr(OcrEngine):
    CHUNK_SIZE = 90

    def __init__(self, s3_bucket: str, s3_file_path: str):
        self.textract_client = boto3.client('textract')
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        self.s3_file_path = s3_file_path

    def ocr(self, image: Image.Image, to_paragraph: bool=False):
        """
        ocr a single image using aws's textract client
        """
        response = self.textract_client.detect_document_text(
            Document={ 'Bytes': image_to_bytes(image) }   
        )
        # to-do:cache the response for later debugging
        if to_paragraph:
            return self.to_paragraph(response)
        return response     
    
    def ocr_batch(self, images: List[Image.Image]):
        """
        Run async OCR on a list or dictionary of images using AWS's Textract client.
        First, upload all the images into an S3 bucket,
        then send async OCR request to Textract with maximum CHUNK_SIZE images.
        Because Textract can have CHUNK_SIZE amount of concurrent request, need to create more than 1 chunk if necessary.
        
        Note: AWS Textract requires input documents to be stored in S3. Therefore, it is necessary to use S3 along with Textract.
        """
        uploaded_images = self._upload_images(images)
        logger.info(f"uploaded {len(uploaded_images)} images to s3")
        chunks = chunk_images(uploaded_images, self.CHUNK_SIZE)
        logger.info(f"split images into {len(chunks)} chunks")
        results = {}
    
        for chunk in chunks:
            try:
                image_to_job_ids = self._start_textract_jobs(chunk)
                job_results = self._wait_for_jobs_and_get_results(image_to_job_ids)
                results.update(job_results)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                logger.error(f"Chunk: {chunk}")
                continue

        return results

    def to_paragraph(self, ocr_response):
        """
        Process OCR result into a paragraph of text (str),
        handling the case where the result is empty or there are errors in the OCR response.
        """
        try:
            if not ocr_response or "Blocks" not in ocr_response:
                return ""
            return "\n".join([item["Text"] for item in ocr_response["Blocks"] if item.get("BlockType") == "LINE"])
        except Exception as e:
            print(f"Error processing OCR response: {e}")
            return ""

        
    
    def _upload_images(self, images: Union[List[Image.Image], Dict[str, Image.Image]]) -> List[str]:
        uploaded_images = []

        if isinstance(images, list):
            for index, image in enumerate(images):
                key = f'{self.s3_file_path}/{str(index)}.jpg'
                self.s3_client.put_object(Bucket=self.s3_bucket, Key=key, Body=image_to_bytes(image))
                uploaded_images.append(f's3://{self.s3_bucket}/{key}')
        elif isinstance(images, dict):
            for file_name, image in images.items():
                key = f'{self.s3_file_path}/{str(file_name)}.jpg'
                self.s3_client.put_object(Bucket=self.s3_bucket, Key=key, Body=image_to_bytes(image))
                uploaded_images.append(f's3://{self.s3_bucket}/{key}')

        return uploaded_images


    def _start_textract_jobs(self, images: List[str]) -> List[str]:
        image_to_job_ids = {}
        for image in images:
            image_path = image.split(f"s3://{self.s3_bucket}/")[1]
            image_basename = os.path.basename(image_path).split(".")[0]
            logger.debug(f"starting textract job for {image_path}")
            response = self.textract_client.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket': self.s3_bucket, 'Name': image_path}})
            #job_ids.append(response['JobId'])
            image_to_job_ids[image_basename] = response['JobId']

        return image_to_job_ids
    
    def _wait_for_jobs_and_get_results(self, image_to_job_ids: Dict[str, str]) -> List[dict]:
        results = {}
        for image_name, job_id in image_to_job_ids.items():
            while True:
                response = self.textract_client.get_document_text_detection(JobId=job_id)
                if response['JobStatus'] in ['SUCCEEDED', 'FAILED']:
                    break
                time.sleep(5)

            if response['JobStatus'] == 'SUCCEEDED':
                results[image_name] = response

        return results