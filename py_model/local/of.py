import json
import os
import sys

import tensorflow as tf
import visualkeras

from exp_input import load_image, load_images

# functions
def load(path):
    model = tf.keras.models.load_model(path)
    return model

# consts
imgs = False
ev = not False
pic = False
n=0
startswith=0

model_path=f"./../../models/model_2_2_3k/py"
dir_path=lambda p: f"{os.getcwd()}/../../data/DataSet{p}"
shape=(224,224)
input_shape=tuple(shape+(3,))
model_settings = {
    "optimizer":tf.keras.optimizers.Adam(learning_rate=0.0003),
    "loss": tf.keras.losses.binary_crossentropy,
    "metrics":['accuracy']
}
model = load(model_path)
model.compile(optimizer=model_settings["optimizer"], loss=model_settings["loss"], metrics=model_settings["metrics"])
print(model.predict(load_image(dir_path('/malignant/Левое бедро.jpg'), newShape=shape)))