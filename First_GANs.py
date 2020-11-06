import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras import backend as K




def main():
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True