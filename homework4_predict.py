# ------------------------------ HOMEWORK ------------------------------------

# Homework 4

from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import argparse

from keras.applications import vgg19
from keras import backend as K

import sys

parser = argparse.ArgumentParser(
    description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')

args = parser.parse_args()
base_image_path = args.base_image_path

# HOMEWORK 4 -----------------------------------------------------------------
model_4 = vgg19.VGG19()
img = load_img(base_image_path, target_size=(224, 224))
img = img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
img = vgg19.preprocess_input(img)
predictions = model_4.predict(img)
label = vgg19.decode_predictions(predictions)
label = label[0][0]
print('%s (%.2f%%)' % (label[1], label[2]*100))
sys.exit()
# HOMEWORK 4 DONE ------------------------------------------------------------
