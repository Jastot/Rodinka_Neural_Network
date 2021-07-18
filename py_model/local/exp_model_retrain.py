import json
import os
import sys

import tensorflow as tf
import tensorflowjs as tfjs

from exp_input import load_image, load_images

# functions
def load(path):
    model = tf.keras.models.load_model(path)
    return model
def train(ds):
    response = model.fit(ds, epochs=train_settings["epochs"])
    return

def save(model, path):
    savedModel = model.save(f"{path}/py")
    savedModeljs = tfjs.converters.save_keras_model(model, f"{path}/js")
    print("saved model")
    return

# consts
startTraining=1
saveModel=True
savePath=f"{os.getcwd()}/../../models/model_2_1_ret"
model_path=f"{os.getcwd()}/../../models/model_2_1/py"
dirPath=lambda p: f"{os.getcwd()}/../../data/DataSet{p}"
shape=(224,224)
inputShape=tuple(shape+(3,))
n=100
startswith=1000
train_settings = {
    "epochs":60,
    "shuffle":True
}
model_settings = {
    "optimizer":tf.keras.optimizers.Adam(learning_rate=0.0003),
    "loss": tf.keras.losses.binary_crossentropy,
    "metrics":['accuracy']
}

# main
model = load(model_path)
model.compile(optimizer=model_settings["optimizer"], loss=model_settings["loss"], metrics=model_settings["metrics"])
model.summary()

input_benign = load_images(dirPath('/benign'), newShape=shape, limit=n, startswith=startswith)
input_malignant = load_images(dirPath('/malignant'), newShape=shape, limit=n, startswith=startswith)
input_tensor = tf.concat([input_benign, input_malignant], axis=0)
output_tensor = tf.one_hot(tf.cast(tf.concat([tf.zeros([input_benign.shape[0]]), tf.ones([input_malignant.shape[0]])], axis=0), 'int32'), depth=2)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(64)


print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")

if(startTraining):
    train(dataset)
    save(model, savePath)