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
dir_path=lambda p: f"{os.getcwd()}/../../data/_data/train{p}"
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
model.summary()
if (pic):
    visualkeras.layered_view(model, to_file=f"{model_path}/../model.png").show()
if (ev):
    benign = load_images(dir_path('/benign'), newShape=shape, limit=n, startswith=startswith)
    malignant = load_images(dir_path('/malignant'), newShape=shape, limit=n, startswith=startswith)
    print(benign.shape[0], malignant.shape[0])
    data = tf.concat([benign, malignant], axis=0)
    labels = tf.one_hot(tf.cast(tf.concat([tf.zeros([benign.shape[0]]), tf.ones([malignant.shape[0]])],axis=0), 'int32'), depth=2)
    print(model.evaluate(data,labels))