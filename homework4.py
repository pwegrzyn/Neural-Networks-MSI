# ------------------------------ HOMEWORK ------------------------------------

# Homework 4

import numpy as np
from keras.preprocessing import image
from keras.applications import vgg19
from keras import backend as K
from PIL import Image
import argparse

parser = argparse.ArgumentParser(
    description='Fooling VGG19 with this one simple trick.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('--object_to_fake', type=int, default=267, required=False,
                    help='Object to fake id.')

args = parser.parse_args()
base_image_path = args.base_image_path
object_type_to_fake = args.object_to_fake

# Prepare model
model = vgg19.VGG19()
model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

# Prepare image
img = image.load_img(base_image_path, target_size=(224, 224))
original_image = image.img_to_array(img)
original_image /= 255.0
original_image -= 0.5
original_image *= 2.0
original_image = np.expand_dims(original_image, axis=0)
adversary_image = np.copy(original_image)

# Setup algorithms
lr = 20.0
loss = model_output_layer[0, object_type_to_fake]
gradient = K.gradients(loss, model_input_layer)[0]
gen = K.function(
    [model_input_layer, K.learning_phase()], [loss, gradient])

# Generate
certainty = 0.0
print("Generating adversary image...")
while certainty < 0.75:
    certainty, gradients = gen([adversary_image, 0])
    adversary_image += gradients * lr
    adversary_image = np.clip(adversary_image, -1.0, 1.0)

# Save result
imamge_res = adversary_image[0]
imamge_res /= 2.0
imamge_res += 0.5
imamge_res *= 255.0
image_res_as_int = imamge_res.astype(np.uint8)
image_to_save = Image.fromarray(image_res_as_int)
image_to_save.save("result/adversary_image.png")
print("Done. Saved to result/adversary_image.png")
