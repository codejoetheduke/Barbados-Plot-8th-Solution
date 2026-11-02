import random
import numpy as np

class Config:
    """Global configuration for the pipeline"""
    RANDOM_SEED = 42
    VAL_SPLIT = 0.1
    IMAGE_SIZE = 1024
    CONF_THRESHOLD = 0.5
    SIMPLIFY_TOLERANCE = 20
    MASK_SIZE = 256  # Zindi evaluation size
    PRETRAINED_MODEL = "CodeJoe/barbados-yolov8m-mapper"
    
    @classmethod
    def set_seed(cls):
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
