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
ev = False
pic = True

model_path=f"{os.getcwd()}/../../models/model_2_1/py"
dir_path=lambda p: f"{os.getcwd()}/../../data/DataSet{p}"
shape=(224,224)
input_shape=tuple(shape+(3,))
model_settings = {
    "optimizer":tf.keras.optimizers.Adam(learning_rate=0.0003),
    "loss": tf.keras.losses.binary_crossentropy,
    "metrics":['accuracy']
}
# main
model = load(model_path)
model.compile(optimizer=model_settings["optimizer"], loss=model_settings["loss"], metrics=model_settings["metrics"])
if (pic):
    visualkeras.layered_view(model, to_file=f"{model_path}/../model.png").show()
if (ev):
    benign = load_images(dir_path('/benign'), newShape=shape)
    malignant = load_images(dir_path('/malignant'), newShape=shape)
    data = tf.concat([benign, malignant], axis=0)
    labels = tf.one_hot(tf.cast(tf.concat([tf.zeros([benign.shape[0]]), tf.ones([malignant.shape[0]])],axis=0), 'int32'), depth=2)
    print(model.evaluate(data,labels))