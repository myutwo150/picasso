import os

base_dir = os.path.dirname(os.path.abspath(__file__))

BACKEND_ML = 'keras'
BACKEND_PREPROCESSOR_NAME = 'preprocess'
BACKEND_PREPROCESSOR_PATH = os.path.join(base_dir, 'util.py')
BACKEND_POSTPROCESSOR_NAME = 'postprocess'
BACKEND_POSTPROCESSOR_PATH = os.path.join(base_dir, 'util.py')
BACKEND_PROB_DECODER_NAME = 'prob_decode'
BACKEND_PROB_DECODER_PATH = os.path.join(base_dir, 'util.py')
BACKEND_TOP_PROBS = 10
DATA_DIR = os.path.join(base_dir, 'data-volume')
