
import keras.applications.mobilenet_v2 as k_mobilenet_v2
import keras.backend as k
import keras.datasets.mnist as k_mnist
import keras.layers as k_layers
import keras.losses as k_losses
import keras.models as k_models
import keras.optimizers as k_optimizers
import keras.preprocessing.image as k_image
import keras.utils as k_utils

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

import tensorflow as tf

import plotly.plotly as py
import plotly.graph_objs as go

mpl.rcParams.update({'figure.max_open_warning': 0})

data = k_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = data
batch_size = 100


def _init_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    init = tf.global_variables_initializer()
    session.run(init)
    return session


def _optimise(session, x, correct_y, loss):
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(loss)
    for i in range(train_images.shape[0] // batch_size):
        batch_train_images = train_images[i *
                                          batch_size:(i + 1) * batch_size, :, :]
        batch_train_labels = train_labels[i * batch_size:(i + 1) * batch_size]
        session.run(train_step, feed_dict={
                    x: batch_train_images, correct_y: batch_train_labels})


def _test_accuracy(session, x, correct_y, y, y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_outcome = session.run(
        accuracy, feed_dict={x: test_images, correct_y: test_labels})
    print("Final accuracy result: {0} %".format(test_outcome*100.0))


# ------------------------------ HOMEWORK ------------------------------------

# Homework 2
def generate_class_activation_map(model, image_path, image_size, kernel_size):
    # Calculate full image result
    image = k_image.load_img(image_path, target_size=(image_size, image_size))
    x = k_image.img_to_array(image)
    x_ = np.expand_dims(x, axis=0)
    x_ = k_mobilenet_v2.preprocess_input(x_)
    predictions = model.predict(x_)
    result = k_mobilenet_v2.decode_predictions(predictions, top=5)[0][0][2]
    # Checking by how much does the result drop by covering all parts of the image one at a time
    heatmap_x = 0
    heatmap_y = 0
    image_orig = k_image.load_img(image_path)
    x_orig = k_image.img_to_array(image_orig)
    height = x_orig.shape[0]
    width = x_orig.shape[1]
    heatmap = np.zeros((height // kernel_size + 1, width // kernel_size + 1))
    for i in range(0, height, kernel_size):
        for j in range(0, width, kernel_size):
            print((heatmap_x, heatmap_y))
            saved_rgb = np.zeros((kernel_size, kernel_size, 3))
            for k in range(0, kernel_size):
                for l in range(0, kernel_size):
                    index_y = i+k if i+k < height else height-1
                    index_x = j+l if j+l < width else width-1
                    for m in range(0, 3):
                        saved_rgb[k][l][m] = x_orig[index_y][index_x][m]
                        x_orig[index_y][index_x][m] = 0.5
            img = k_image.array_to_img(x_orig)
            img = img.resize((224, 224))
            arr = k_image.img_to_array(img)
            arr_ = np.expand_dims(arr, axis=0)
            arr_ = k_mobilenet_v2.preprocess_input(arr_)
            predictions = model.predict(arr_)
            diff = result - \
                k_mobilenet_v2.decode_predictions(predictions, top=5)[0][0][2]
            heatmap[heatmap_y][heatmap_x] = diff if diff > 0 else 0
            for k in range(0, kernel_size):
                for l in range(0, kernel_size):
                    index_y = i+k if i+k < height else height-1
                    index_x = j+l if j+l < width else width-1
                    for m in range(0, 3):
                        x_orig[index_y][index_x][m] = saved_rgb[k][l][m]
            heatmap_x += 1
        heatmap_x = 0
        heatmap_y += 1
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.show()


def test_cam():
    model = k_mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)
    image_path = 'img/doberman.jpg'
    generate_class_activation_map(model, image_path, 224, 5)


def main():
    test_cam()


if __name__ == '__main__':
    main()
