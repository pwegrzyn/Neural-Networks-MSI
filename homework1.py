
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

# Homework 1
def visualize_activations(model, x, start=2, end=5):
    if len(model.layers) <= start or len(model.layers) <= end:
        print('Layer bounds are invalid!')
        return
    activations = []
    layer_names = []
    for i in range(start, end):
        get_activations = k.function([model.layers[0].input], [
                                     model.layers[i].output])
        activation = get_activations([x])
        activations.append(activation[0])
        layer_names.append(model.layers[i].name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std() if channel_image.std() != 0 else 1
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


def test_visualization():
    model = k_mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    print("Network containts following layers:")
    for i, (name, layer) in enumerate(layers.items()):
        print("Layer {0} : {1}".format(i, (name, layer)))
    print("Together: {0} parameters\n".format(model.count_params()))
    image_path = 'img/nosacz.jpg'
    image = k_image.load_img(image_path, target_size=(224, 224))
    x = k_image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = k_mobilenet_v2.preprocess_input(x)
    visualize_activations(model, x, start=2, end=5)


def main():
    test_visualization()


if __name__ == '__main__':
    main()
