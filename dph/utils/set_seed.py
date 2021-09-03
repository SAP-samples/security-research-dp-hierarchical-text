import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed_value):
    # See here: https://github.com/NVIDIA/tensorflow-determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
