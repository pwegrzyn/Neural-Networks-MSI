# ------------------------------ HOMEWORK ------------------------------------

# Homework 4

from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import argparse

from keras.applications import xception
from keras.applications import vgg19
from keras.applications import mobilenet_v2
from keras.applications import inception_v3
from keras.applications import densenet
from keras import backend as K

import sys


# HOMEWORK 4 -----------------------------------------------------------------
model = xception.Xception(weights='imagenet', include_top=False)
layers = dict([(layer.name, layer.output) for layer in model.layers])
print("Network containts following layers:")
for i, (name, layer) in enumerate(layers.items()):
        print("Layer {0} : {1}".format(i, (name, layer)))
# HOMEWORK 4 DONE ------------------------------------------------------------
