from joblib import dump, load
import os



current_file_path = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_file_path, "models", "ocr_tree.joblib")
ocr_feature_names = ['jaccard', 'jaccard_letter', 'frame_token_ct',
       'word_dis', 'letter_dissim'] #'dissimilarity', 
ocr_new_slide_clf = load(MODEL_PATH)    

PHASH_THRESHOLD = 10
PHASH_PERCENTILE = 0.95

DURATION_THRESHOLD = 3 
WORD_CT_THRESHOLD = 5

DHASH_THRESHOLD = 10
DHASH_PERCENTILE = 0.95

YOUTUBE_VIDEO_DOWNLOAD_PATH = os.environ.get("YOUTUBE_VIDEO_DOWNLOAD_PATH", "./youtube_downloads")
SLIDE_EXTRACT_OUTPUT_BUCKET_NAME = os.environ.get("SLIDE_EXTRACT_OUTPUT_BUCKET_NAME", "slide-extract-output")
VALID_YOUTUBE_CHANNELS = ['PyData', 'Open Data Science', 'Kaggle', 'AWS Online Tech Talks', 'Microsoft Ignite', 'Microsoft']
MAX_VIDEO_DURATION = 3800


