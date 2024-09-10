import tf2onnx
import onnx
import tensorflow as tf
from onnx2pytorch import ConvertModel

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Convolution2D,
    ZeroPadding2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Activation,
)

import os
path = os.path.join(os.getcwd(), 'deepface\\deepface\\weights\\')
def load_model_weight(name: str):
    # abs_path = path + name
    rel_path = os.path.join(path, name)
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f"No such file or directory: '{rel_path}'")
    model = vgg_model()
    model.load_weights(rel_path)
    return model

def convert_model(model: tf.keras.Model, name: str):
    onnx_model = tf2onnx.convert.from_keras(model)
    onnx.save(onnx_model, f"{path}{name}.onnx")
    pytorch_model = ConvertModel(f"{path}{name}.onnx")
    pytorch_model.save_model(f"{path}{name}.pth")

def vgg_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model

convert_model(load_model_weight("vgg_face_weights.h5"), "vgg_face_weights")
# print(os.getcwd())
# loaded_model = tf.keras.models.load_model('vgg_face_weights.h5')