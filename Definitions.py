import os
import multiprocessing

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
DATA_RAW_DIR = os.path.join(ROOT_DIR, 'Data/raw')
DATA_TRAIN_DIR = os.path.join(ROOT_DIR, 'Data/train')
DATA_VALIDATION_DIR = os.path.join(ROOT_DIR, 'Data/validation')

#CPU_NUMBER = multiprocessing.cpu_count()
CPU_NUMBER = 1
